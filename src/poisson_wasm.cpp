#include <stdint.h>
#include <stdlib.h>
#include <emscripten.h>
#include <emscripten/html5.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

static void main_loop() {
  EM_ASM_({ draw(); });
}

int main() {
  emscripten_set_canvas_element_size("#canvas", g_width, g_height);
  emscripten_set_main_loop(main_loop, 0, 1);
}
