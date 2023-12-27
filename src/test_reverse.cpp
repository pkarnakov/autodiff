#include <cmath>
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
  auto x = Tracer<T>(var_x);
  auto y = Tracer<T>(var_y);
  PE(x);
  PE((sin(x) * cos(y) + cos(x) * sin(y)));
}

int main() {
  TestReverse();
}
