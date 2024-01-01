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
  template <class Node>
  void Visit(Node* node) {
    dot.Write(node);
  }
  DotWriter dot;
};

struct ExtraPrint : public BaseExtra {
  template <class Node>
  void Visit(Node* node) {
    std::cout << *node << std::endl;
  }
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
    const auto order = e.GetFowardOrder();
    e.UpdateGrad(order);
    PE(e.value());
    PE(x.grad());
    PE(y.grad());
    std::cout << path << std::endl;
    std::ofstream fout(path);
    Extra extra(fout);
    Traverse(order, extra);
    ClearVisited(order);
  };
  auto e1 = sum(sin(x) * cos(y) + cos(x) * sin(y));
  eval(e1, "reverse_" + suff + "1.gv");
  auto e2 = sum(sin(x + y));
  eval(e2, "reverse_" + suff + "2.gv");
}

template <class T = double>
static void TestRoll() {
  std::cout << '\n' << __func__ << std::endl;
  auto Str = [](auto m) { return MatrixToStr(m, 3); };
  const auto matr = Matrix<T>::iota(5);
  const auto zeros = Matrix<T>::zeros_like(matr);
  Var var_x(zeros, "x");
  PEN(Str(matr));
  auto x = MakeTracer<Extra>(var_x);
  auto grad = [&](auto e) {
    e.UpdateGrad();
    return x.grad();
  };
  PEN(Str(grad(sum(x * matr))));
  PEN(Str(grad(sum(roll(x, 1, 2) * matr))));
}

template <class T = double>
static void TestMultigrid() {
  std::cout << '\n' << __func__ << std::endl;
  auto Str = [](auto m) { return MatrixToStr(m, 5); };
  const auto ufine = Matrix<T>::iota(10) * 2;
  const auto u = ufine.restrict();
  auto zeros = Matrix<T>::zeros_like(u);
  Var var_x(zeros, "x");
  PEN(Str(u));
  PEN(Str(ufine));
  auto x = MakeTracer<Extra>(var_x);
  auto grad = [&](auto e) {
    e.UpdateGrad();
    return x.grad();
  };
  PEN(Str(grad(sum(x * u))));
  PEN(Str(grad(sum(interpolate(x) * ufine) / 4)));
}

int main() {
  TestReverse(1., "scal");
  TestReverse(Matrix<double>::eye(3), "matr");
  TestRoll();
  TestMultigrid();
}
