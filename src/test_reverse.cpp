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

void TestNested() {
  std::cout << '\n' << __func__ << std::endl;
  Var<double> var_x(0, "x");
  Var<double> var_y(0, "y");
  Var<double> var_one(1, "1");
  Tracer<double> x(var_x);
  Tracer<double> y(var_y);
  Tracer<double> one(var_one);
  Var<Tracer<double>> var_tx(x, one, "tx");
  Var<Tracer<double>> var_ty(y, one, "ty");
  Tracer<Tracer<double>> tx(var_tx);
  Tracer<Tracer<double>> ty(var_ty);
  auto e = sin(tx) * cos(ty) + cos(tx) * sin(ty);
  PrintDot("nested_value.gv", e.value());
  PrintDot("nested_grad.gv", e.grad());
}

int main() {
  TestReverse();
  TestNested();
}
