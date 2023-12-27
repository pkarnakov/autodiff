#include <cmath>
#include <fstream>
#include <iostream>

#include "reverse.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << std::endl;

void TestReverse() {
  using T = double;
  const auto pi = M_PI;
  std::cout << '\n' << __func__ << std::endl;
  Var<T> var_x(pi / 6, "x");
  Var<T> var_y(pi / 3, "y");
  PE(var_x);
  PE(var_y);
  Tracer<T> x(var_x);
  Tracer<T> y(var_y);
  auto e = sin(x) * cos(y) + cos(x) * sin(y);
  PE(e);
  PrintDot("graph.gv", e);
}

int main() {
  TestReverse();
}
