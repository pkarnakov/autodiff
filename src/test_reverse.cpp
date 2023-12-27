#include <cmath>
#include <iostream>

#include "reverse.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << std::endl;

void TestReverse() {
  using T = double;
  std::cout << '\n' << __func__ << std::endl;
  Var<T> var_x(1, "x");
  Var<T> var_y(2, "y");
  std::cout << var_x << ' ' << var_y << std::endl;
  auto x = Tracer<T>(var_x);
  auto y = Tracer<T>(var_y);
  auto e = x;
  std::cout << e.value() << ' ' << e.grad() << std::endl;
}

int main() {
  TestReverse();
}
