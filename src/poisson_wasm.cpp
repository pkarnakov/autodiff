#include <emscripten.h>
#include <emscripten/html5.h>
#include <stdint.h>
#include <stdlib.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
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
  int width = 800;
  int height = 800;
  int Nx = 128;
  int epochs_per_frame = 25;
  int max_nlvl = 4;
  Scal lr = 1e-3;
  Scal uref_k = 2;
  std::array<Scal, 2> ulim = {-1, 1};

  // Control flags.
  bool is_pause = false;

  // Solver state.
  Matrix<Scal> uref;  // Reference solution.
  Matrix<Scal> rhs;   // Right-hand side.
  std::vector<std::unique_ptr<Var<M>>> var_uu;  // Multigrid components.
  std::vector<Tracer<M, Extra>> uu;  // Tracers of multigrid components.
  Tracer<M, Extra> u;                // Sum of multigrid components.
  NodeOrder<Extra> order;
  Tracer<Scal, Extra> loss;

  // Optimizer.
  std::unique_ptr<Adam> optimizer;
  typename Adam::Config opt_config;
  std::function<void(int)> callback;
  std::function<void()> update_grads;
  std::vector<M*> opt_vars;
  std::vector<const M*> opt_grads;
  std::chrono::time_point<Clock> time_prev;

  // Data buffers.
  std::vector<Vect> particles;
  std::vector<uint32_t> bitmap;

  // Visualization.
};

std::shared_ptr<Scene> g_scene;

static void InitScene(Scene& scene) {
  // Init particles.
  const int n = 1000;
  scene.particles.resize(n);
  Scal phi = 0;
  for (size_t i = 0; i < scene.particles.size(); ++i) {
    using std::cos;
    using std::sin;
    const Scal r = 0.05 + phi * 0.01;
    phi += 10 / (n * r);
    scene.particles[i] = {0.5 + r * cos(phi), 0.5 + r * sin(phi)};
  }

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
  int nx = Nx;
  auto& var_uu = scene.var_uu;
  auto& uu = scene.uu;
  // Create variables for multigrid levels.
  for (int i = 0; i < scene.max_nlvl; ++i) {
    const auto name = "u" + std::to_string(i + 1);
    var_uu.emplace_back(std::make_unique<Var<M>>(M::zeros(nx), name));
    uu.emplace_back(*var_uu.back());
    nx /= 2;
    if (nx <= 4) {
      break;
    }
  }
  std::cout << "multigrid levels: ";
  for (auto& var_u : var_uu) {
    auto& m = var_u->value();
    std::cout << "(" << m.nrow() << "," << m.ncol() << ") ";
  }
  std::cout << std::endl;

  auto u = uu.back();
  for (size_t i = uu.size() - 1; i > 0;) {
    --i;
    u = interpolate(u) + uu[i];
  }
  scene.loss = mean(sqr(eval_lapl(u) - scene.rhs));
  scene.order = scene.loss.GetFowardOrder();
  scene.u = u;

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
      printf(
          "epoch=%5d, loss=%8.6e, throughput=%.3fM cells/s, u:[%.3f,%.3f], \n",
          epoch, scene.loss.value(), throughput, scene.u.value().min(),
          scene.u.value().max());
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
  scene.optimizer->Run(scene.opt_config, scene.opt_vars, scene.opt_grads,
                       scene.update_grads, scene.callback);
  auto& particles = scene.particles;
  for (size_t i = 0; i < particles.size(); ++i) {
    auto& p = particles[i];
    p[0] += 0.02;
    p[1] += 0.01;
    if (p[0] > 1) {
      p[0] -= 1;
    }
    if (p[1] > 1) {
      p[1] -= 1;
    }
  }
  const auto& u = scene.u.value();
  scene.bitmap.resize(u.size());
  const auto& ulim = scene.ulim;
  const Scal c0[] = {0.85, 0.5, 0.07};
  const Scal c1[] = {0.97, 0.97, 0.97};
  const Scal c2[] = {0.34, 0.17, 0.54};
  auto linear = [&](Scal x, Scal x0, Scal x1, Scal u0, Scal u1) {
    return ((x1 - x) * u0 + (x - x0) * u1) / (x1 - x0);
  };
  auto color = [&](Scal v, int k) -> uint8_t {
    return 255 * (v < 0.5 ? linear(v, 0, 0.5, c0[k], c1[k])
                          : linear(v, 0.5, 1, c1[k], c2[k]));
  };
  for (size_t i = 0; i < u.size(); ++i) {
    const Scal v = std::max<Scal>(
        0, std::min<Scal>(1, (u[i] - ulim[0]) / (ulim[1] - ulim[0])));
    const uint8_t r = color(v, 0);
    const uint8_t g = color(v, 1);
    const uint8_t b = color(v, 2);
    scene.bitmap[i] = (0xff << 24) | (b << 16) | (g << 8) | r;
  }
  CopyToCanvas(scene.bitmap.data(), u.ncol(), u.nrow());
}

extern "C" {
int GetParticles(uint16_t* data, int max_size) {
  const int entrysize = 2;
  int i = 0;
  auto& scene = *g_scene;
  const auto& particles = scene.particles;
  for (auto p : particles) {
    if (i + entrysize > max_size) {
      break;
    }
    data[i + 0] = p[0] * scene.width;
    data[i + 1] = p[1] * scene.height;
    i += entrysize;
  }
  return i;
}
void SendKeyDown(char keysym) {
  (void)keysym;
}
int GetBitmapWidth() {
  return g_scene->Nx;
}

int GetBitmapHeight() {
  return g_scene->Nx;
}

void SendMouseMotion(Scal x, Scal y) {
  (void)x;
  (void)y;
}

void SendMouseDown(Scal x, Scal y) {
  (void)x;
  (void)y;
}

void SendMouseUp(Scal x, Scal y) {
  (void)x;
  (void)y;
}

void Init() {
  g_scene = std::make_shared<Scene>();
  auto& scene = *g_scene;

  InitScene(scene);
}

void SetPause(int flag) {
  g_scene->is_pause = flag;
}

const char* GetMouseMode() {
  return "mode";
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
