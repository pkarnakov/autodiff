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
using Matr = Matrix<Scal>;
using Extra = BaseExtra;
using Clock = std::chrono::steady_clock;
using Adam = optimizer::Adam<Scal>;

struct Scene {
  int Nx = 128;
  std::array<Scal, 2> xlim = {0, 1};
  std::array<Scal, 2> ylim = {0, 1};
  int epochs_per_frame = 10;
  int max_nlvl = 4;
  Scal max_vel = 10;  // Maximum velocity.
  Scal kimp = 10;    // Weight of imposed values.
  Scal kreg = 0.005;  // Weight of velocity regularization.
  Scal kregu = 0.001;  // Weight of field regularization.
  Scal lr = 0.005;
  std::array<Scal, 2> ulim = {-0.5, 0.5};

  // Solver state.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_u;   // Solution.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_vx;  // Velocity in x.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_vy;  // Velocity in y.
  std::unique_ptr<Var<Matr>> var_mask;               // Mask.
  Tracer<Matr, Extra> mask;
  NodeOrder<Extra> order;
  Tracer<Scal, Extra> loss;
  std::function<Matr(const Matr&)> transform_vx;
  std::function<Matr(const Matr&)> transform_vy;

  // Optimizer.
  std::unique_ptr<Adam> optimizer;
  typename Adam::Config opt_config;
  std::function<void(int)> callback;
  std::function<void()> update_grads;
  std::vector<Matr*> opt_vars;
  std::vector<const Matr*> opt_grads;
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
  const size_t Ny = Nx;
  const auto& xlim = scene.xlim;
  const auto& ylim = scene.ylim;
  const Scal hx = (xlim[1] - xlim[0]) / Nx;
  const Scal hy = (ylim[1] - ylim[0]) / Ny;

  auto inner = Matr::zeros(Nx);
  // Exclude edges.
  for (size_t ix = 2; ix + 2 < Nx; ++ix) {
    for (size_t iy = 2; iy + 2 < Ny; ++iy) {
      inner(ix, iy) = 1;
    }
  }

  auto transform_vx = [&max_vel = scene.max_vel](auto& vx) {  //
    return tanh(vx) * max_vel;
  };
  auto transform_vy = [](auto& vy) {  //
    return 0.5 + sigmoid(vy);
  };
  scene.transform_vx = transform_vx;
  scene.transform_vy = transform_vy;

  auto fd_second = [](auto& u, int dx, int dy, Scal h) {
    auto upp = roll(u, -dx * 2, -dy * 2);
    auto up = roll(u, -dx, -dy);
    return (-upp + 4 * up - 3 * u) * (0.5 / h);
  };
  auto fd_first = [](auto& u, int dx, int dy, Scal h) {
    auto up = roll(u, -dx, -dy);
    return (up - u) / h;
  };
  auto fd_central = [](auto& u, int dx, int dy, Scal h) {
    auto up = roll(u, -dx, -dy);
    auto um = roll(u, dx, dy);
    return (up - um) * (0.5 / h);
  };
  (void)fd_first;

  // Evaluates the discrete operator.
  auto operator_advection = [hx, hy, fd_central, fd_second](auto& u, auto& vx,
                                                            auto& vy) {
    if (0) {  // Second order upwind.
      auto u_xm = fd_second(u, -1, 0, -hx);
      auto u_xp = fd_second(u, 1, 0, hx);
      auto u_ym = fd_second(u, 0, -1, -hy);
      auto u_yp = fd_second(u, 0, 1, hy);
      auto q_x = maximum(vx, 0) * u_xm + minimum(vx, 0) * u_xp;
      // auto q_y = maximum(vy, 0) * u_ym + minimum(vy, 0) * u_yp;
      auto q_y = fd_central(u, 0, 1, hy) * vy;
      return q_x + q_y;
    }
    // Central.
    auto q_x = fd_central(u, 1, 0, hx) * vx;
    auto q_y = fd_central(u, 0, 1, hy) * vy;
    return q_x + q_y;
  };

  // Evaluates Laplacian.
  auto operator_lapl = [hx](auto& u) {  //
    const Scal hxx = sqr(hx);
    const Scal a = -4 / hxx;
    const Scal b = 1 / hxx;
    return conv(u, a, b, b, b, b);
  };

  // Create variable for solution.
  scene.mg_u = std::make_unique<MultigridVar<Scal, Extra>>(  //
      Matr::zeros(Nx), scene.max_nlvl, "u");
  // Create variables for velocity.
  scene.mg_vx = std::make_unique<MultigridVar<Scal, Extra>>(  //
      Matr::zeros(Nx), scene.max_nlvl, "vx");
  scene.mg_vy = std::make_unique<MultigridVar<Scal, Extra>>(  //
      Matr::zeros(Nx), scene.max_nlvl, "vy");
  // Create variable for mask.
  scene.var_mask = std::make_unique<Var<Matr>>(Matr::zeros(Nx), "mask");
  scene.mask = MakeTracer<Extra>(*scene.var_mask);

  {  // Compute loss.
    auto& u = scene.mg_u->tracer();
    auto& mask = scene.mask;
    auto vx = transform_vx(scene.mg_vx->tracer());
    auto vy = transform_vy(scene.mg_vy->tracer());
    scene.loss = mean(sqr(operator_advection(u, vx, vy) * inner)) +
                 mean(sqr((u - mask) * scene.kimp)) +
                 mean(sqr(operator_lapl(u) * (inner * scene.kregu))) +
                 mean(sqr(operator_lapl(vx) * (inner * scene.kreg))) +
                 mean(sqr(operator_lapl(vy) * (inner * scene.kreg)));
  }

  scene.time_prev = Clock::now();
  scene.callback = [&scene](int epoch) {
    if (epoch % scene.epochs_per_frame == 0) {
      auto time_curr = Clock::now();
      auto delta = time_curr - scene.time_prev;
      scene.time_prev = time_curr;
      auto msdur = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
      auto ms = msdur.count();
      const Matr& u = scene.mg_u->tracer().value();
      const Matr vx = scene.transform_vx(scene.mg_vx->tracer().value());
      const Matr vy = scene.transform_vy(scene.mg_vy->tracer().value());
      const Scal throughput =
          (ms > 0 ? 1e-3 * u.size() * scene.epochs_per_frame / ms : 0);
      auto print = [&](char* buf, size_t bufsize) -> int {
        return std::snprintf(  //
            buf, bufsize,
            "epoch=%5d, loss=%.4e, vx=%.3f, vy=%.3f,<br>"
            "throughput=%.3fM cells/s",
            epoch, std::sqrt(scene.loss.value()), vx.mean(), vy.mean(),
            throughput);
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
  scene.mg_vx->AppendValues(scene.opt_vars);
  scene.mg_vx->AppendGrads(scene.opt_grads);
  scene.mg_vy->AppendValues(scene.opt_vars);
  scene.mg_vy->AppendGrads(scene.opt_grads);
  scene.optimizer = std::make_unique<Adam>();
}

static void AdvanceScene(Scene& scene) {
  if (scene.is_pause) {
    return;
  }
  scene.optimizer->Run(scene.opt_config, scene.opt_vars, scene.opt_grads,
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
