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
  Scal kimp = 10;    // Weight of imposed values.
  Scal kreg = 0.005;    // Weight of line field regularization.
  Scal kregu = 0.001;   // Weight of field regularization.
  Scal kunit = 10;      // Weight of penalty on deviation of |q| from 1.
  Scal q_init = 0.1;    // Amplitude of random initial line field.
  Scal q_eps = 1e-6;    // Regularization of |q| at zero.
  Scal lr = 0.005;
  std::array<Scal, 2> ulim = {-0.5, 0.5};

  // Solver state.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_u;   // Solution.
  // Line field q = |q| * (cos(2*theta), sin(2*theta)) of direction
  // n = (cos(theta), sin(theta)), invariant to n -> -n, so lines have
  // no orientation. Penalty on deviation of |q| from 1 excludes the trivial
  // solution q = 0 but allows |q| = 0 at defects, e.g. the center of a circle.
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_qa;
  std::unique_ptr<MultigridVar<Scal, Extra>> mg_qb;
  std::unique_ptr<Var<Matr>> var_mask;               // Mask.
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
  std::vector<float> segments;  // Endpoints of segments showing line field.

  // Control and visualization.
  Scal circle_radius = 3;  // Radius of circle drawn on the mask on mouse click.
  bool is_pause = false;
  bool show_lines = true;  // Show the line field as segments.
  int lines_step = 8;      // Spacing of segments in cells.
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

  auto fd_central = [](auto& u, int dx, int dy, Scal h) {
    auto up = roll(u, -dx, -dy);
    auto um = roll(u, dx, dy);
    return (up - um) * (0.5 / h);
  };

  // Evaluates |q| * (n . grad u)^2, the squared residual of advection
  // along the lines, from components q = (qa, qb) and their norm qn.
  // Nonnegative since (ux^2 - uy^2, 2 ux uy) has norm |grad u|^2.
  auto operator_line = [hx, hy, fd_central](auto& u, auto& qa, auto& qb,
                                            auto& qn) {
    auto ux = fd_central(u, 1, 0, hx);
    auto uy = fd_central(u, 0, 1, hy);
    auto uxx = sqr(ux);
    auto uyy = sqr(uy);
    return (qn * (uxx + uyy) + qa * (uxx - uyy) + qb * (ux * uy * 2)) * 0.5;
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
  // Create variables for line field.
  scene.mg_qa = std::make_unique<MultigridVar<Scal, Extra>>(  //
      Matr::zeros(Nx), scene.max_nlvl, "qa");
  scene.mg_qb = std::make_unique<MultigridVar<Scal, Extra>>(  //
      Matr::zeros(Nx), scene.max_nlvl, "qb");
  {  // Initialize the coarsest level with random values,
     // since away from the reference solution q = 0 is a stationary point.
    std::mt19937 gen(0);
    std::uniform_real_distribution<Scal> dist(-scene.q_init, scene.q_init);
    for (auto* mg : {scene.mg_qa.get(), scene.mg_qb.get()}) {
      auto& v = mg->vars().back()->value();
      for (size_t i = 0; i < v.size(); ++i) {
        v[i] = dist(gen);
      }
    }
  }
  // Create variable for mask.
  scene.var_mask = std::make_unique<Var<Matr>>(Matr::zeros(Nx), "mask");
  scene.mask = MakeTracer<Extra>(*scene.var_mask);

  {  // Compute loss.
    auto& u = scene.mg_u->tracer();
    auto& mask = scene.mask;
    auto& qa = scene.mg_qa->tracer();
    auto& qb = scene.mg_qb->tracer();
    auto qn = sqrt(sqr(qa) + sqr(qb) + scene.q_eps);
    scene.loss = mean(operator_line(u, qa, qb, qn) * inner) +
                 mean(sqr((u - mask) * scene.kimp)) +
                 mean(sqr(operator_lapl(u) * (inner * scene.kregu))) +
                 mean(sqr(operator_lapl(qa) * (inner * scene.kreg))) +
                 mean(sqr(operator_lapl(qb) * (inner * scene.kreg))) +
                 mean(sqr((1 - qn) * scene.kunit));
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
      const Matr& qa = scene.mg_qa->tracer().value();
      const Matr& qb = scene.mg_qb->tracer().value();
      const Scal qmag = sqrt(sqr(qa) + sqr(qb)).mean();
      const Scal throughput =
          (ms > 0 ? 1e-3 * u.size() * scene.epochs_per_frame / ms : 0);
      auto print = [&](char* buf, size_t bufsize) -> int {
        return std::snprintf(  //
            buf, bufsize,
            "epoch=%5d, loss=%.4e, |q|=%.3f,<br>"
            "throughput=%.3fM cells/s",
            epoch, std::sqrt(scene.loss.value()), qmag, throughput);
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
  scene.mg_qa->AppendValues(scene.opt_vars);
  scene.mg_qa->AppendGrads(scene.opt_grads);
  scene.mg_qb->AppendValues(scene.opt_vars);
  scene.mg_qb->AppendGrads(scene.opt_grads);
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

// Draws the line field on the canvas as segments of direction
// n = (cos(theta), sin(theta)) with length proportional to min(|q|, 1).
static void DrawLines(Scene& scene) {
  const Matr& qa = scene.mg_qa->tracer().value();
  const Matr& qb = scene.mg_qb->tracer().value();
  const size_t nx = qa.nrow();
  const size_t ny = qa.ncol();
  const size_t step = scene.lines_step;
  auto& seg = scene.segments;
  seg.clear();
  for (size_t ix = step / 2; ix < nx; ix += step) {
    for (size_t iy = step / 2; iy < ny; iy += step) {
      const Scal a = qa(ix, iy);
      const Scal b = qb(ix, iy);
      const Scal theta = 0.5 * std::atan2(b, a);
      const Scal len = std::min<Scal>(std::sqrt(a * a + b * b), 1);
      // Half-length in fractions, a bit shorter than the spacing.
      const Scal hlx = 0.4 * len * step / nx * std::cos(theta);
      const Scal hly = 0.4 * len * step / ny * std::sin(theta);
      const Scal x = (ix + 0.5) / nx;
      const Scal y = (iy + 0.5) / ny;
      seg.insert(seg.end(), {x - hlx, y - hly, x + hlx, y + hly});
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
  if (keysym == 'l') {
    g_scene->show_lines = !g_scene->show_lines;
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
  if (g_scene->show_lines) {
    DrawLines(*g_scene);
  }
}

int main() {
  Init();
  emscripten_set_main_loop(main_loop, 0, 1);
}
