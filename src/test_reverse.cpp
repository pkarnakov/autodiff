#include <cmath>
#include <fstream>
#include <iostream>

#include "matrix.h"
#include "reverse.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << std::endl;

template <class T = double>
static void TestReverse() {
  const auto pi = M_PI;
  std::cout << '\n' << __func__ << std::endl;
  auto var_x = Var<T>(pi / 6, "x");
  auto var_y = Var<T>(pi / 3, "y");
  PE(var_x);
  PE(var_y);
  auto x = Tracer(var_x);
  auto y = Tracer(var_y);
  auto e = sin(x) * cos(y) + cos(x) * sin(y);
  PE(e);
  PrintDot("graph.gv", e);
}

struct Extra {
  template <class Node>
  static void Apply(Node* node) {
    std::cout << *node << '\n';
  }
};

template <class T = double>
static void TestGrad() {
  std::cout << '\n' << __func__ << std::endl;
  const auto eye = Matrix<T>::eye(3, 3);
  const auto zeros = Matrix<T>::zeros(3, 3);
  using M = Matrix<T>;
  auto var_x = Var<M>(eye, "x");
  auto var_y = Var<M>(1 + eye, "y");
  PE(var_x);
  PE(var_y);
  auto x = MakeTracer<Extra>(var_x);
  auto y = MakeTracer<Extra>(var_y);
  auto e = sum(sin(x) * cos(y) + cos(x) * sin(y));
  e.UpdateGrad(1.);
  PE(e.value());
  PE(x.grad());
  PE(y.grad());
  e.Apply();
}

template <class T = double>
static void TestNested() {
  std::cout << '\n' << __func__ << std::endl;
  auto var_x = Var<T>(0, "x");
  auto var_y = Var<T>(0, "y");
  auto var_one = Var<T>(1, "1");
  auto x = Tracer(var_x);
  auto y = Tracer(var_y);
  auto one = Tracer(var_one);
  auto var_tx = Var(x, one, "tx");
  auto var_ty = Var(y, one, "ty");
  auto tx = Tracer(var_tx);
  auto ty = Tracer(var_ty);
  auto e = sin(tx) * cos(ty) + cos(tx) * sin(ty);
  PrintDot("nested_value.gv", e.value());
  PrintDot("nested_grad.gv", e.grad());
}

int main() {
  // TestReverse();
  TestGrad();
  // TestNested();
}
