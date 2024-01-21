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
  std::array<Scal, 2> xlim = {-1, 1};
  std::array<Scal, 2> tlim = {0, 1};
  int epochs_per_frame = 20;
  int max_nlvl = 4;
  Scal lr = 0.005;
  std::array<Scal, 2> ulim = {-0.5, 0.5};

  // Solver state.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_u;  // Solution.
  M u_ref;                                          // Reference solution.
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
  const size_t Nt = Nx;
  const auto& xlim = scene.xlim;
  const auto& tlim = scene.tlim;
  const Scal hx = (xlim[1] - xlim[0]) / Nx;
  const Scal ht = (tlim[1] - tlim[0]) / Nt;

  {  // Reference solution.
    auto u_ref = M::zeros(Nx);
    for (size_t ix = 0; ix < Nx; ++ix) {
      for (size_t it = 0; it < Nt; ++it) {
        using std::cos;
        const Scal x = xlim[0] + (ix + 0.5) * hx;
        const Scal t = tlim[0] + (it + 0.5) * ht;
        const Scal pi = M_PI;
        Scal u = 0;
        for (int k = 1; k <= 5; ++k) {
          u += cos((x - t + 0.5) * k * pi);
          u += cos((x + t - 0.5) * k * pi);
        }
        u_ref(ix, it) = u / 10;
      }
    }
    scene.u_ref = u_ref;
  }

  auto inner = M::zeros(Nx);
  for (size_t ix = 1; ix + 1 < Nx; ++ix) {
    for (size_t it = 1; it + 1 < Nt; ++it) {
      inner(ix, it) = 1;
    }
  }

  auto transform_u = [inner](auto& u) {  //
    return u;
  };
  scene.transform_u = transform_u;

  // Evaluates the discrete operator.
  auto operator_wave = [hx, ht, &transform_u](auto& u) {
    const Scal hxx = sqr(hx);
    const Scal htt = sqr(ht);
    const Scal a = 2 / hxx - 2 / htt;
    const Scal ax = -1 / hxx;
    const Scal at = 1 / htt;
    return conv(transform_u(u), a, ax, ax, at, at);
  };

  // Initial guess.
  auto u_init = M::zeros(Nx);
  // Create variable for solution.
  scene.mg_u =
      std::make_unique<MultigridVar<Scal, Extra>>(u_init, scene.max_nlvl, "u");
  // Create variable for mask.
  scene.var_mask = std::make_unique<Var<M>>(M::zeros(Nx), "mask");
  scene.mask = Tracer<M, Extra>(*scene.var_mask);

  {  // Compute loss.
    auto& u = scene.mg_u->tracer();
    auto& mask = scene.mask;
    auto& u_ref = scene.u_ref;
    scene.loss = mean(sqr(operator_wave(u) * inner)) +
                 mean(sqr((u - u_ref) * mask / sqr(hx)));
  }

  scene.time_prev = Clock::now();
  scene.callback = [&scene](int epoch) {
    if (epoch % scene.epochs_per_frame == 0) {
      auto time_curr = Clock::now();
      auto delta = time_curr - scene.time_prev;
      scene.time_prev = time_curr;
      auto msdur = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
      auto ms = msdur.count();
      const auto& u = scene.mg_u->tracer().value();
      const Scal throughput =
          (ms > 0 ? 1e-3 * u.size() * scene.epochs_per_frame / ms : 0);
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
