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

#include "matrix.h"
#include "optimizer.h"
#include "reverse.h"

using Scal = double;
using Vect = std::array<Scal, 2>;
using M = Matrix<Scal>;
using Extra = BaseExtra;
using Clock = std::chrono::steady_clock;
using Adam = optimizer::Adam<Scal>;

struct Scene {
  int width = 512;
  int height = 512;
  int Nx = 128;
  int epochs_per_frame = 25;
  int max_nlvl = 4;
  Scal lr = 1e-3;
  Scal uref_k = 2;
  std::array<Scal, 2> ulim = {-0.8, 0.8};

  // Control flags.
  bool is_pause = false;
  bool is_mouse_down = false;

  // Solver state.
  Matrix<Scal> uref;                            // Reference solution.
  Matrix<Scal> rhs;                             // Right-hand side.
  std::vector<std::unique_ptr<Var<M>>> var_uu;  // Multigrid components.
  std::vector<Tracer<M, Extra>> uu;  // Tracers of multigrid components.
  Tracer<M, Extra> u;                // Sum of multigrid components.
  NodeOrder<Extra> order;
  Tracer<Scal, Extra> loss;
  std::unique_ptr<Var<M>> var_mask;
  Tracer<M, Extra> mask;

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

  // Visualization.
  std::vector<char> status_string = {0};
};

int Select(Scal u, int min, int max) {
  return std::min<int>(max,
                       std::max<int>(min, std::round((1 - u) * min + u * max)));
}

template <class T>
T Clip(T u, T min, T max) {
  return std::min<T>(max, std::max<T>(min, u));
}

std::shared_ptr<Scene> g_scene;

static void InitScene(Scene& scene) {
  const size_t Nx = scene.Nx;
  const Scal hx = 1. / Nx;

  // Reference solution.
  auto uref = Matrix<Scal>::zeros(Nx);
  for (size_t i = 0; i < Nx; ++i) {
    for (size_t j = 0; j < Nx; ++j) {
      using std::sin;
      const Scal x = (i + 0.5) / Nx;
      const Scal y = (j + 0.5) / Nx;
      const Scal pi = M_PI;

      const Scal k = scene.uref_k;
      uref(i, j) = sin(pi * sqr(k * x)) * sin(pi * y);
    }
  }
  scene.uref = uref;
  auto eval_lapl = [hx](auto& u) {  //
    return conv(u, -4, 1, 1, 1, 1) / sqr(hx);
  };

  scene.rhs = eval_lapl(uref);
  auto& var_uu = scene.var_uu;
  auto& uu = scene.uu;
  {  // Create variables for multigrid levels.
    int nx = Nx;
    for (int i = 0; i < scene.max_nlvl; ++i) {
      const auto name = "u" + std::to_string(i + 1);
      var_uu.emplace_back(std::make_unique<Var<M>>(M::zeros(nx), name));
      uu.emplace_back(*var_uu.back());
      nx /= 2;
      if (nx <= 4) {
        break;
      }
    }
    auto u = uu.back();
    for (size_t i = uu.size() - 1; i > 0;) {
      --i;
      u = interpolate(u) + uu[i];
    }
    scene.u = u;
  }
  // Create variable for mask.
  scene.var_mask = std::make_unique<Var<M>>(M::zeros(Nx), "mask");
  scene.mask = Tracer<M, Extra>(*scene.var_mask);
  scene.loss = mean(sqr(eval_lapl(scene.u) - scene.rhs)) +
               mean(sqr(scene.u * scene.mask / sqr(hx)));
  scene.order = scene.loss.GetFowardOrder();

  scene.time_prev = Clock::now();

  scene.callback = [&scene](int epoch) {
    if (epoch % scene.epochs_per_frame == 0) {
      auto time_curr = Clock::now();
      auto delta = time_curr - scene.time_prev;
      scene.time_prev = time_curr;
      auto msdur = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
      auto ms = msdur.count();
      const Scal throughput =
          (ms > 0 ? 1e-3 * scene.rhs.size() * scene.epochs_per_frame / ms : 0);
      auto print = [&](char* buf, size_t bufsize) -> int {
        return std::snprintf(
            buf, bufsize,
            "epoch=%5d, loss=%.4e<br>throughput=%.3fM cells/s, u:[%.3f,%.3f]",
            epoch, scene.loss.value(), throughput, scene.u.value().min(),
            scene.u.value().max());
      };
      auto& s = scene.status_string;
      s.resize(print(nullptr, 0) + 1);
      print(s.data(), s.size());
    }
  };

  scene.update_grads = [&scene]() { scene.loss.UpdateGrad(scene.order); };
  scene.opt_config.epochs = scene.epochs_per_frame;
  scene.opt_config.lr = scene.lr;
  for (size_t i = 0; i < var_uu.size(); ++i) {
    scene.opt_vars.push_back(&var_uu[i]->value());
    scene.opt_grads.push_back(&uu[i].grad());
  }
  scene.optimizer = std::make_unique<Adam>();
}

void CopyToCanvas(uint32_t* buf, int w, int h) {
  EM_ASM_(
      {
        let data = Module.HEAPU8.slice($0, $0 + $1 * $2 * 4);
        let tctx = g_tmp_canvas.getContext("2d");
        let image = tctx.getImageData(0, 0, $1, $2);
        image.data.set(data);
        tctx.putImageData(image, 0, 0);
      },
      buf, w, h);
}

void UpdateScene() {
  auto& scene = *g_scene;
  if (scene.is_pause) {
    return;
  }
  {
    /*
    auto& state = scene.optimizer->state();
    // Clear optimizer state, to adapt to changing mask.
    state.vv.clear();
    state.mm.clear();
    */
  }
  scene.optimizer->Run(scene.opt_config, scene.opt_vars, scene.opt_grads,
                       scene.update_grads, scene.callback);
  const auto& u = scene.u.value();
  const auto& mask = scene.mask.value();
  scene.bitmap.resize(u.size());
  const auto& ulim = scene.ulim;
  auto linear = [](Scal x, Scal x0, Scal x1, Scal u0, Scal u1) {
    return ((x1 - x) * u0 + (x - x0) * u1) / (x1 - x0);
  };
  const Scal c3[] = {0, 0.95, 0.6};  // Green.
  auto color = [linear](Scal v, int k) {
    const Scal c0[] = {0.34, 0.17, 0.54};  // Blue.
    const Scal c1[] = {0.97, 0.97, 0.97};  // White.
    const Scal c2[] = {0.85, 0.5, 0.07};   // Orange.
    return (v < 0.5 ? linear(v, 0, 0.5, c0[k], c1[k])
                    : linear(v, 0.5, 1, c1[k], c2[k]));
  };
  auto blend = [](Scal c, Scal ca, Scal a) {  //
    return c * (1 - a) + ca * a;
  };
  auto round = [](Scal c) -> uint8_t {
    return std::max<Scal>(0, std::min<Scal>(255, std::round(c * 255)));
  };
  for (size_t i = 0; i < u.nrow(); ++i) {
    for (size_t j = 0; j < u.ncol(); ++j) {
      const Scal vu = std::max<Scal>(
          0, std::min<Scal>(1, (u(i, j) - ulim[0]) / (ulim[1] - ulim[0])));
      const Scal vmask = mask(i, j);
      const uint8_t r = round(blend(color(vu, 0), c3[0], vmask * 0.5));
      const uint8_t g = round(blend(color(vu, 1), c3[1], vmask * 0.5));
      const uint8_t b = round(blend(color(vu, 2), c3[2], vmask * 0.5));
      scene.bitmap[j * u.ncol() + i] = (0xff << 24) | (b << 16) | (g << 8) | r;
    }
  }
  CopyToCanvas(scene.bitmap.data(), u.ncol(), u.nrow());
}

void DrawCircle(Scal x, Scal y) {
  auto& scene = *g_scene;
  auto& mask = scene.var_mask->value();
  const int ic = Select(x, 0, mask.nrow() - 1);
  const int jc = Select(1 - y, 0, mask.ncol() - 1);
  const int w = 3;
  auto kernel = [](int dx, int dy) -> Scal {  //
    return Clip<Scal>(0, 1, 2 * (1 - std::sqrt(sqr(dx) + sqr(dy)) / w));
  };
  const int i0 = Clip<int>(ic - w, 0, mask.nrow());
  const int i1 = Clip<int>(ic + w, 0, mask.nrow());
  const int j0 = Clip<int>(jc - w, 0, mask.ncol());
  const int j1 = Clip<int>(jc + w, 0, mask.ncol());
  for (int i = i0; i < i1; ++i) {
    for (int j = j0; j < j1; ++j) {
      const Scal a = kernel(i - ic, j - jc);
      mask(i, j) = Clip<Scal>(a + mask(i, j) * (1 - a), 0, 1);
    }
  }
}

void ResetOptimizer() {
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
  DrawCircle(x, y);
}

void SendMouseDown(Scal x, Scal y) {
  auto& scene = *g_scene;
  scene.is_mouse_down = true;
  DrawCircle(x, y);
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
  UpdateScene();
  EM_ASM({ draw(); });
}

int main() {
  Init();
  auto& scene = *g_scene;
  emscripten_set_canvas_element_size("#canvas", scene.width, scene.height);
  emscripten_set_main_loop(main_loop, 0, 1);
}
