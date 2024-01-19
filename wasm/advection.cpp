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
  std::array<Scal, 2> xlim = {0, 1};
  std::array<Scal, 2> tlim = {0, 1};
  Scal max_vel = 4;  // Maximum velocity.
  int epochs_per_frame = 20;
  int max_nlvl = 4;
  Scal lr = 0.005;
  std::array<Scal, 2> ulim = {-0.7, 0.7};

  // Solver state.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_u;  // Solution.
  NodeOrder<Extra> order;
  Tracer<Scal, Extra> loss;
  std::unique_ptr<Var<M>> var_mask;
  Tracer<M, Extra> mask;
  std::unique_ptr<Var<Scal>> var_vel;
  Tracer<Scal, Extra> vel;
  std::function<Scal(const Scal&)> transform_vel;

  // Optimizer.
  std::unique_ptr<Adam> optimizer;
  typename Adam::Config opt_config;
  std::function<void(int)> callback;
  std::function<void()> update_grads;
  std::vector<M*> opt_vars;
  std::vector<const M*> opt_grads;
  std::vector<Scal*> opt_vars_scal;
  std::vector<const Scal*> opt_grads_scal;
  std::chrono::time_point<Clock> time_prev;

  // Data buffers.
  std::vector<uint32_t> bitmap;

  // Control and visualization.
  Scal circle_radius = 3;  // Radius of circle drawn on the mask on mouse click.
  bool is_pause = false;
  bool is_mouse_down = false;
  std::vector<char> status_string = {0};  // Null-terminated string.
};

std::shared_ptr<Scene> g_scene;

static void InitScene(Scene& scene) {
  const size_t Nx = scene.Nx;
  const size_t Nt = Nx;
  const auto& xlim = scene.xlim;
  const auto& tlim = scene.tlim;
  const Scal hx = (xlim[1] - xlim[0]) / Nx;
  const Scal ht = (tlim[1] - tlim[0]) / Nt;

  auto inner = M::zeros(Nx);
  // Exclude both edges in x and lower edge in t.
  for (size_t ix = 2; ix + 2 < Nx; ++ix) {
    for (size_t it = 1; it + 1 < Nt; ++it) {
      inner(ix, it) = 1;
    }
  }

  auto transform_vel = [&max_vel = scene.max_vel](auto& vel) {
    return tanh(vel) * max_vel;
  };
  scene.transform_vel = transform_vel;

  // Evaluates the discrete operator.
  auto operator_advection = [hx, ht](auto& u, auto& vel) {
    auto uxmm = roll(u, 2, 0);
    auto uxm = roll(u, 1, 0);
    auto uxpp = roll(u, -2, 0);
    auto uxp = roll(u, -1, 0);
    auto utp = roll(u, 0, -1);
    auto utm = roll(u, 0, 1);
    auto u_t = (utp - utm) * (0.5 / ht);
    auto u_xm = (3 * u - 4 * uxm + uxmm) * (0.5 / hx);
    auto u_xp = (-3 * u + 4 * uxp - uxpp) * (0.5 / hx);
    auto q_x = maximum(vel, 0) * u_xm + minimum(vel, 0) * u_xp;
    return u_t + q_x;
  };

  // Initial guess.
  auto u_init = M::zeros(Nx);
  // Create variable for solution.
  scene.mg_u =
      std::make_unique<MultigridVar<Scal, Extra>>(u_init, scene.max_nlvl, "u");
  // Create variable for mask.
  scene.var_mask = std::make_unique<Var<M>>(M::zeros(Nx), "mask");
  scene.mask = MakeTracer<Extra>(*scene.var_mask);
  // Create variable for velocity.
  scene.var_vel = std::make_unique<Var<Scal>>(0, "vel");
  scene.vel = MakeTracer<Extra>(*scene.var_vel);

  {  // Compute loss.
    auto& u = scene.mg_u->tracer();
    auto& mask = scene.mask;
    auto vel = transform_vel(scene.vel);
    scene.loss = mean(sqr(operator_advection(u, vel) * inner)) +  //
                 mean(sqr(u - mask) / hx);
  }

  scene.time_prev = Clock::now();
  scene.callback = [&scene](int epoch) {
    if (epoch % scene.epochs_per_frame == 0) {
      auto time_curr = Clock::now();
      auto delta = time_curr - scene.time_prev;
      scene.time_prev = time_curr;
      auto msdur = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
      auto ms = msdur.count();
      const M& u = scene.mg_u->tracer().value();
      const Scal vel = scene.transform_vel(scene.vel.value());
      const Scal throughput =
          (ms > 0 ? 1e-3 * u.size() * scene.epochs_per_frame / ms : 0);
      auto print = [&](char* buf, size_t bufsize) -> int {
        return std::snprintf(  //
            buf, bufsize,
            "epoch=%5d, loss=%.4e, vel=%.3f,<br>"
            "throughput=%.3fM cells/s",
            epoch, std::sqrt(scene.loss.value()), vel, throughput);
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
  scene.opt_vars_scal.push_back(&scene.var_vel->value());
  scene.opt_grads_scal.push_back(&scene.vel.grad());
  scene.optimizer = std::make_unique<Adam>();
}

static void AdvanceScene(Scene& scene) {
  if (scene.is_pause) {
    return;
  }
  scene.optimizer->Run(scene.opt_config, scene.opt_vars, scene.opt_grads,
                       scene.opt_vars_scal, scene.opt_grads_scal,
                       scene.update_grads, scene.callback);
  const auto& u = scene.mg_u->tracer().value();
  const Matrix<float> unorm =
      (u - scene.ulim[0]) / (scene.ulim[1] - scene.ulim[0]);
  DrawFieldsOnBitmap(unorm, scene.mask.value(), scene.bitmap);
  CopyToCanvas(scene.bitmap.data(), u.ncol(), u.nrow());
}

extern "C" {
int GetBitmapWidth() {
  return g_scene->Nx;
}

int GetBitmapHeight() {
  return g_scene->Nx;
}

void SendKeyDown(char keysym) {
  (void)keysym;
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
  InitScene(*g_scene);
}

void SetPause(int flag) {
  g_scene->is_pause = flag;
}

char* GetStatusString() {
  return g_scene->status_string.data();
}
}  // extern "C"

static void main_loop() {
  AdvanceScene(*g_scene);
  EM_ASM({ draw(); });
}

int main() {
  Init();
  emscripten_set_main_loop(main_loop, 0, 1);
}
