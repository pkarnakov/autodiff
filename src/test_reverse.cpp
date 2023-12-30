#include <cmath>
#include <fstream>
#include <iostream>

#include "matrix.h"
#include "reverse.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << std::endl;

struct Extra : public BaseExtra {
  Extra(std::ostream& out) : dot(out) {}
  template <class N>
  void TraversePre(N* node) {
    dot.Write(node);
  }
  DotWriter dot;
};

template <class T>
static void TestReverse(const T& eye, std::string suff) {
  std::cout << '\n' << __func__ << std::endl;
  const auto pi = M_PI;
  Var var_x(eye * (pi / 8), "x");
  Var var_y(eye * (pi / 8), "y");
  PE(var_x);
  PE(var_y);
  auto x = MakeTracer<Extra>(var_x);
  auto y = MakeTracer<Extra>(var_y);
  auto eval = [&](auto& e, std::string path) {
    e.ClearGrad();
    e.UpdateGrad();
    PE(e.value());
    PE(x.grad());
    PE(y.grad());
    std::cout << path << std::endl;
    std::ofstream fout(path);
    Extra extra(fout);
    e.TraversePre(extra);
  };
  auto e1 = sum(sin(x) * cos(y) + cos(x) * sin(y));
  eval(e1, "reverse_" + suff + ".gv");
  auto e2 = sum(sin(x + y));
  eval(e2, "reverse_" + suff + ".gv");
}

template <class T = double>
static void TestNested() {
  // TODO: Implement.
  std::cout << '\n' << __func__ << std::endl;
  Var<T> var_x(0, "x");
  Var<T> var_y(0, "y");
  auto x = MakeTracer<Extra>(var_x);
  auto y = MakeTracer<Extra>(var_y);
  Var var_tx(x, "tx");
  Var var_ty(y, "ty");
  auto tx = MakeTracer<Extra>(var_tx);
  auto ty = MakeTracer<Extra>(var_ty);
  auto e = sin(tx) * cos(ty) + cos(tx) * sin(ty);
  auto eval = [&](auto& tracer, std::string path) {
    std::cout << path << std::endl;
    std::ofstream fout(path);
    Extra extra(fout);
    tracer.TraversePre(extra);
  };
  // eval(e.value(), "nested_value.gv");
  // eval(e.grad(), "nested_grad.gv");
}

template <class T = double>
static void TestRoll() {
  std::cout << '\n' << __func__ << std::endl;
  auto Str = [](auto m) { return MatrixToStr(m, 3); };
  auto matr = Matrix<T>::iota(5);
  auto zeros = Matrix<T>::zeros_like(matr);
  Var var_x(zeros, "x");
  PEN(Str(matr));
  auto x = MakeTracer<Extra>(var_x);
  auto grad = [&](auto e) {
    e.ClearGrad();
    e.UpdateGrad();
    return x.grad();
  };
  PEN(Str(grad(sum(x * matr))));
  PEN(Str(grad(sum(roll(x, 1, 2) * matr))));
}

int main() {
  TestReverse(1., "scal");
  TestReverse(Matrix<double>::eye(3), "matr");
  TestRoll();
  // TestNested();
}
