#include <emscripten.h>
#include <emscripten/html5.h>
#include <stdint.h>
#include <stdlib.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using Scal = double;
using Vect = std::array<Scal, 2>;

struct SceneData {
  std::vector<Vect> particles;
};

const int g_width = 800;
const int g_height = 800;
SceneData g_data;
bool g_is_pause;

void UpdateScene() {
  if (g_is_pause) {
    return;
  }
  for (size_t i = 0; i < g_data.particles.size(); ++i) {
    auto& p = g_data.particles[i];
    p[0] += 0.02;
    p[1] += 0.01;
    if (p[0] > 1) {
      p[0] -= 1;
    }
    if (p[1] > 1) {
      p[1] -= 1;
    }
  }
}

extern "C" {
int GetParticles(uint16_t* data, int max_size) {
  const int entrysize = 2;
  int i = 0;
  for (auto p : g_data.particles) {
    if (i + entrysize > max_size) {
      break;
    }
    data[i + 0] = p[0] * g_width;
    data[i + 1] = p[1] * g_height;
    i += entrysize;
  }
  return i;
}
void SendKeyDown(char keysym) {
}
void SendMouseMotion(Scal x, Scal y) {
}
void SendMouseDown(Scal x, Scal y) {
}
void SendMouseUp(Scal x, Scal y) {
}
void Init() {
  g_is_pause = false;

  const int n = 1000;
  g_data.particles.resize(n);
  Scal phi = 0;
  for (size_t i = 0; i < g_data.particles.size(); ++i) {
    using std::sin;
    using std::cos;
    const Scal r = 0.05 + phi * 0.01;
    phi += 10 / (n * r);
    g_data.particles[i] = {0.5 + r * cos(phi), 0.5 + r * sin(phi)};
  }
}
void SetPause(int flag) {
  g_is_pause = flag;
}
const char* GetMouseMode() {
  return "mode";
}
} // extern "C"

static void main_loop() {
  UpdateScene();
  // NULL added to suppress warning -Wgnu-zero-variadic-macro-arguments.
  EM_ASM({ draw(); }, NULL);
}

int main() {
  Init();
  emscripten_set_canvas_element_size("#canvas", g_width, g_height);
  emscripten_set_main_loop(main_loop, 0, 1);
}
