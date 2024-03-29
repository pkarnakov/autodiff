#include <emscripten.h>
#include <emscripten/html5.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "graphics.h"
#include "matrix.h"
#include "optimizer.h"
#include "reverse.h"

using Scal = float;
using Vect = std::array<Scal, 2>;
using M = Matrix<Scal>;
using Extra = BaseExtra;
using Clock = std::chrono::steady_clock;
using Adam = optimizer::Adam<Scal>;

struct Scene {
  int Nx = 128;
  int epochs_per_frame = 20;
  int max_nlvl = 4;
  Scal lr = 0.005;
  Scal osc_k = 2;
  std::array<Scal, 2> ulim = {-0.8, 0.8};

  // Solver state.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_u;  // Solution.
  M u_ref;                                          // Reference solution.
  M rhs;                                            // Right-hand side.
  NodeOrder<Extra> order;
  Tracer<Scal, Extra> loss;
  std::unique_ptr<Var<M>> var_mask;
  Tracer<M, Extra> mask;
  std::function<M(const M&)> transform_u;

  // Optimizer.
  std::unique_ptr<Adam> optimizer;
  typename Adam::Config opt_config;
  std::function<void(int)> callback;
  std::function<void()> update_grads;
  std::vector<M*> opt_vars;
  std::vector<const M*> opt_grads;
  std::chrono::time_point<Clock> time_prev;

  // Data buffers.
  std::vector<uint32_t> bitmap;

  // Control and visualization.
  Scal circle_radius = 3;  // Radius of circle drawn on the mask on mouse click.
  bool is_pause = false;
  bool is_mouse_down = false;
  std::vector<char> status_string = {0};
};

std::shared_ptr<Scene> g_scene;

static void InitScene(Scene& scene) {
  const size_t Nx = scene.Nx;
  const Scal hx = 1. / Nx;

  {  // Reference solution.
    auto u_ref = M::zeros(Nx);
    for (size_t i = 0; i < Nx; ++i) {
      for (size_t j = 0; j < Nx; ++j) {
        using std::sin;
        const Scal x = (i + 0.5) / Nx;
        const Scal y = (j + 0.5) / Nx;
        const Scal pi = M_PI;

        const Scal k = scene.osc_k;
        u_ref(i, j) = sin(pi * sqr(k * x)) * sin(pi * y);
      }
    }
    scene.u_ref = u_ref;
  }

  auto inner = M::zeros(Nx);
  for (size_t i = 1; i + 1 < Nx; ++i) {
    for (size_t j = 1; j + 1 < Nx; ++j) {
      inner(i, j) = 1;
    }
  }

  auto transform_u = [inner](auto& u) {  //
    return u * inner;
  };
  scene.transform_u = transform_u;

  // Evaluates Laplacian.
  auto operator_lapl = [hx, &transform_u](auto& u) {  //
    const Scal hxx = sqr(hx);
    const Scal a = -4 / hxx;
    const Scal b = 1 / hxx;
    return conv(transform_u(u), a, b, b, b, b);
  };

  // Initial guess.
  auto u_init = M::zeros(Nx);
  // Right-hand side.
  scene.rhs = operator_lapl(scene.u_ref);
  // Create variable for solution.
  scene.mg_u =
      std::make_unique<MultigridVar<Scal, Extra>>(u_init, scene.max_nlvl, "u");
  // Create variable for mask.
  scene.var_mask = std::make_unique<Var<M>>(M::zeros(Nx), "mask");
  scene.mask = Tracer<M, Extra>(*scene.var_mask);

  {  // Compute loss.
    auto& rhs = scene.rhs;
    auto& u = scene.mg_u->tracer();
    auto& mask = scene.mask;
    scene.loss = mean(sqr(operator_lapl(u) - rhs)) +  //
                 mean(sqr(u * mask / sqr(hx)));
  }

  scene.time_prev = Clock::now();
  scene.callback = [&scene](int epoch) {
    if (epoch % scene.epochs_per_frame == 0) {
      auto& u = scene.mg_u->tracer().value();
      auto time_curr = Clock::now();
      auto delta = time_curr - scene.time_prev;
      scene.time_prev = time_curr;
      auto msdur = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
      auto ms = msdur.count();
      const Scal throughput =
          (ms > 0 ? 1e-3 * scene.rhs.size() * scene.epochs_per_frame / ms : 0);
      auto print = [&](char* buf, size_t bufsize) -> int {
        return std::snprintf(  //
            buf, bufsize,
            "epoch=%5d, loss=%.4e,<br>"
            "throughput=%.3fM cells/s, u:[%.3f,%.3f]",
            epoch, std::sqrt(scene.loss.value()), throughput, u.min(), u.max());
      };
      auto& s = scene.status_string;
      s.resize(print(nullptr, 0) + 1);
      print(s.data(), s.size());
    }
  };

  scene.order = scene.loss.GetFowardOrder();
  scene.update_grads = [&scene]() {  //
    scene.loss.UpdateGrad(scene.order);
  };
  scene.opt_config.epochs = scene.epochs_per_frame;
  scene.opt_config.lr = scene.lr;
  scene.mg_u->AppendValues(scene.opt_vars);
  scene.mg_u->AppendGrads(scene.opt_grads);
  scene.optimizer = std::make_unique<Adam>();
}

static void AdvanceScene() {
  auto& scene = *g_scene;
  if (scene.is_pause) {
    return;
  }
  scene.optimizer->Run(scene.opt_config, scene.opt_vars, scene.opt_grads,
                       scene.update_grads, scene.callback);
  const auto& u = scene.transform_u(scene.mg_u->tracer().value());
  const Matrix<float> unorm =
      (u - scene.ulim[0]) / (scene.ulim[1] - scene.ulim[0]);
  DrawFieldsOnBitmap(unorm, scene.mask.value(), scene.bitmap);
  CopyToCanvas(scene.bitmap.data(), u.ncol(), u.nrow());
}

static void ResetOptimizer() {
  auto& scene = *g_scene;
  auto& state = scene.optimizer->state();
  // Clear optimizer state, to adapt to changing mask.
  state.vv.clear();
  state.mm.clear();
}

extern "C" {
int GetBitmapWidth() {
  return g_scene->Nx;
}

int GetBitmapHeight() {
  return g_scene->Nx;
}

void SendKeyDown(char keysym) {
  if (keysym == 'o') {
    ResetOptimizer();
  }
}

void SendMouseMotion(Scal x, Scal y) {
  auto& scene = *g_scene;
  if (!scene.is_mouse_down) {
    return;
  }
  DrawCircleOnMask(scene.var_mask->value(), x, y, scene.circle_radius);
}

void SendMouseDown(Scal x, Scal y) {
  auto& scene = *g_scene;
  scene.is_mouse_down = true;
  DrawCircleOnMask(scene.var_mask->value(), x, y, scene.circle_radius);
}

void SendMouseUp(Scal x, Scal y) {
  (void)x;
  (void)y;
  auto& scene = *g_scene;
  scene.is_mouse_down = false;
}

void Init() {
  g_scene = std::make_shared<Scene>();
  auto& scene = *g_scene;

  InitScene(scene);
}

void SetPause(int flag) {
  g_scene->is_pause = flag;
}

char* GetStatusString() {
  return g_scene->status_string.data();
}
}  // extern "C"

static void main_loop() {
  AdvanceScene();
  EM_ASM({ draw(); });
}

int main() {
  Init();
  emscripten_set_main_loop(main_loop, 0, 1);
}
