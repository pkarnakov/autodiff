#include <emscripten.h>
#include <emscripten/html5.h>

#include <algorithm>
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
#include <random>
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
  Scal kimp = 10;       // Weight of imposed values.
  Scal kreg = 0.1;      // Weight of velocity regularization (gradient).
  Scal kregu = 0.001;   // Weight of field regularization.
  Scal kunit = 3;       // Weight of penalty on deviation of |v| from 1.
  Scal kzero = 1;       // Weight of penalty on |v| where it is not required.
  Scal grad_ref = 0.1;  // Change of u per cell at which |v| = 1 is required.
  Scal vel_init = 0.1;  // Amplitude of random initial velocity.
  Scal lr = 0.005;
  std::array<Scal, 2> ulim = {-0.5, 0.5};

  // Solver state.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_u;  // Solution.
  // Velocity. Required to have |v| = 1 where u changes, with weight w,
  // and pulled to zero elsewhere, with weight 1 - w.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_vx;  // Velocity in x.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_vy;  // Velocity in y.
  std::unique_ptr<Var<Matr>> var_mask;               // Mask.
  // Weight w in [0, 1] of the requirement |v| = 1, a function of |grad u|.
  // Updated from u every frame and not differentiated.
  std::unique_ptr<Var<Matr>> var_w;
  Tracer<Matr, Extra> mask;
  NodeOrder<Extra> order;
  Tracer<Scal, Extra> loss;

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
  std::vector<float> segments;  // Endpoints of segments showing velocity.

  // Control and visualization.
  Scal circle_radius = 3;  // Radius of circle drawn on the mask on mouse click.
  bool is_pause = false;
  bool show_arrows = true;  // Show the velocity as arrows.
  int arrows_step = 8;      // Spacing of arrows in cells.
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

  // Second-order one-sided difference in direction (dx, dy) with step h.
  auto fd_second = [](auto& u, int dx, int dy, Scal h) {
    auto upp = roll(u, -dx * 2, -dy * 2);
    auto up = roll(u, -dx, -dy);
    return (-upp + 4 * up - 3 * u) * (0.5 / h);
  };

  // Evaluates the discrete operator v . grad u with second-order upwind
  // differences, so values propagate downstream only.
  auto operator_advection = [hx, hy, fd_second](auto& u, auto& vx, auto& vy) {
    auto u_xm = fd_second(u, -1, 0, -hx);
    auto u_xp = fd_second(u, 1, 0, hx);
    auto u_ym = fd_second(u, 0, -1, -hy);
    auto u_yp = fd_second(u, 0, 1, hy);
    auto q_x = maximum(vx, 0) * u_xm + minimum(vx, 0) * u_xp;
    auto q_y = maximum(vy, 0) * u_ym + minimum(vy, 0) * u_yp;
    return q_x + q_y;
  };

  // Evaluates |grad u|^2 with forward differences.
  auto operator_grad2 = [hx, hy](auto& u) {
    auto u_x = (roll(u, -1, 0) - u) / hx;
    auto u_y = (roll(u, 0, -1) - u) / hy;
    return sqr(u_x) + sqr(u_y);
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
  {  // Initialize the coarsest level with random values,
     // since away from the reference solution v = 0 is a stationary point.
    std::mt19937 gen(0);
    std::uniform_real_distribution<Scal> dist(-scene.vel_init, scene.vel_init);
    for (auto* mg : {scene.mg_vx.get(), scene.mg_vy.get()}) {
      auto& v = mg->vars().back()->value();
      for (size_t i = 0; i < v.size(); ++i) {
        v[i] = dist(gen);
      }
    }
  }
  // Create variable for mask.
  scene.var_mask = std::make_unique<Var<Matr>>(Matr::zeros(Nx), "mask");
  scene.mask = MakeTracer<Extra>(*scene.var_mask);
  scene.var_w = std::make_unique<Var<Matr>>(Matr::zeros(Nx), "w");

  {  // Compute loss.
    auto& u = scene.mg_u->tracer();
    auto& mask = scene.mask;
    auto& vx = scene.mg_vx->tracer();
    auto& vy = scene.mg_vy->tracer();
    auto w = MakeTracer<Extra>(*scene.var_w);
    auto v2 = sqr(vx) + sqr(vy);
    scene.loss = mean(sqr(operator_advection(u, vx, vy) * inner)) +
                 mean(sqr((u - mask) * scene.kimp)) +
                 mean(sqr(operator_lapl(u) * (inner * scene.kregu))) +
                 mean(operator_grad2(vx) * (inner * sqr(scene.kreg))) +
                 mean(operator_grad2(vy) * (inner * sqr(scene.kreg))) +
                 mean(sqr(1 - v2) * w * sqr(scene.kunit)) +
                 mean(v2 * (1 - w) * sqr(scene.kzero));
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
      const Matr& vx = scene.mg_vx->tracer().value();
      const Matr& vy = scene.mg_vy->tracer().value();
      const Scal vmag = sqrt(sqr(vx) + sqr(vy)).mean();
      const Scal throughput =
          (ms > 0 ? 1e-3 * u.size() * scene.epochs_per_frame / ms : 0);
      auto print = [&](char* buf, size_t bufsize) -> int {
        return std::snprintf(  //
            buf, bufsize,
            "epoch=%5d, loss=%.4e, |v|=%.3f,<br>"
            "throughput=%.3fM cells/s",
            epoch, std::sqrt(scene.loss.value()), vmag, throughput);
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

// Updates the weight of the requirement |v| = 1 from the current solution.
static void UpdateWeight(Scene& scene) {
  const Matr& u = scene.mg_u->tracer().value();
  // Central differences in units of change per cell.
  const Matr du_x = (u.roll(-1, 0) - u.roll(1, 0)) * 0.5;
  const Matr du_y = (u.roll(0, -1) - u.roll(0, 1)) * 0.5;
  const Matr g2 = (sqr(du_x) + sqr(du_y)) / sqr(scene.grad_ref);
  scene.var_w->value() = 1 - exp(-g2);
}

static void AdvanceScene(Scene& scene) {
  if (scene.is_pause) {
    return;
  }
  scene.optimizer->Run(scene.opt_config, scene.opt_vars, scene.opt_grads,
                       scene.update_grads, scene.callback);
  // After Run(), since values of the solution are computed there.
  UpdateWeight(scene);
  const auto& u = scene.mg_u->tracer().value();
  const Matrix<float> unorm =
      (u - scene.ulim[0]) / (scene.ulim[1] - scene.ulim[0]);
  DrawFieldsOnBitmap(unorm, scene.mask.value(), scene.bitmap);
  CopyToCanvas(scene.bitmap.data(), u.ncol(), u.nrow());
}

// Draws the velocity on the canvas as arrows
// with length proportional to |v|.
static void DrawArrows(Scene& scene) {
  const Matr& vx = scene.mg_vx->tracer().value();
  const Matr& vy = scene.mg_vy->tracer().value();
  const size_t nx = vx.nrow();
  const size_t ny = vx.ncol();
  const size_t step = scene.arrows_step;
  const Scal head = 0.35;         // Length of head relative to arrow.
  const Scal cs = std::cos(0.5);  // Half-angle of head.
  const Scal sn = std::sin(0.5);
  auto& seg = scene.segments;
  seg.clear();
  for (size_t ix = step / 2; ix < nx; ix += step) {
    for (size_t iy = step / 2; iy < ny; iy += step) {
      const Scal a = vx(ix, iy);
      const Scal b = vy(ix, iy);
      const Scal norm = std::sqrt(a * a + b * b);
      if (norm == 0) {
        continue;
      }
      const Scal len = norm;
      // Half of the arrow in fractions, a bit shorter than the spacing.
      const Scal hlx = 0.4 * len * step / nx * a / norm;
      const Scal hly = 0.4 * len * step / ny * b / norm;
      const Scal x = (ix + 0.5) / nx;
      const Scal y = (iy + 0.5) / ny;
      const Scal tx = x + hlx;
      const Scal ty = y + hly;
      // Head: shaft reversed and rotated by +-0.5 radians.
      const Scal bx = -2 * hlx * head;
      const Scal by = -2 * hly * head;
      seg.insert(seg.end(), {x - hlx, y - hly, tx, ty});
      seg.insert(seg.end(),
                 {tx, ty, tx + cs * bx - sn * by, ty + sn * bx + cs * by});
      seg.insert(seg.end(),
                 {tx, ty, tx + cs * bx + sn * by, ty - sn * bx + cs * by});
    }
  }
  DrawSegmentsOnCanvas(seg);
}

extern "C" {
int GetBitmapWidth() {
  return g_scene->Nx;
}

int GetBitmapHeight() {
  return g_scene->Nx;
}

void SendKeyDown(char keysym) {
  if (keysym == 'a') {
    g_scene->show_arrows = !g_scene->show_arrows;
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
  if (g_scene->show_arrows) {
    DrawArrows(*g_scene);
  }
}

int main() {
  Init();
  emscripten_set_main_loop(main_loop, 0, 1);
}
