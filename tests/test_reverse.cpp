#include <cmath>
#include <fstream>
#include <iostream>

#include "matrix.h"
#include "reverse.h"

// Print and evaluate.
#define PN(a) std::cout << '\n' << (a) << std::endl;
#define PE(a) std::cout << #a << ": " << (a) << std::endl;
#define PEN(a) std::cout << #a << ":\n" << (a) << std::endl;

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

template <class Scal = double>
static void TestMatrix() {
  std::cout << '\n' << __func__ << std::endl;
  auto Str = [](auto m) { return MatrixToStr(m, 3); };
  const auto matr = Matrix<Scal>::iota(5);
  Var var_u(Matrix<Scal>::ones_like(matr) * 2, "u");
  Var var_x(Scal(5), "x");
  auto u = MakeTracer<Extra>(var_u);
  auto x = MakeTracer<Extra>(var_x);
  auto grad_u = [&](auto e) {
    e.UpdateGrad();
    return '\n' + Str(u.grad());
  };
  auto grad_x = [&](auto e) {
    e.UpdateGrad();
    return x.grad();
  };
  PEN(Str(matr));
  PEN(Str(u.value()));
  PE(x.value());
  PN("Roll.");
  PE(grad_u(sum(u * matr)));
  PE(grad_u(sum(roll(u, 1, 2) * matr)));
  PN("Element access.");
  PE(grad_u(u(1, 2)));
  PN("Tracer scalar by constant matrix.");
  PE(grad_x(sum(x * matr)));
  PE(grad_x(sum(matr * x)));
  PN("Tracer scalar by tracer matrix.");
  PE(grad_u(sum(x * u)));
  PE(grad_x(sum(x * u)));
  PE(grad_u(sum(u * x)));
  PE(grad_x(sum(u * x)));
  PE(grad_u(sum(x + u)));
  PE(grad_x(sum(x + u)));
  PE(grad_u(sum(u + x)));
  PE(grad_x(sum(u + x)));
  PE(grad_u(sum(x - u)));
  PE(grad_x(sum(x - u)));
  PE(grad_u(sum(u - x)));
  PE(grad_x(sum(u - x)));
  PE(grad_u(sum(u / x)));
  PE(grad_x(sum(u / x)));
  PN("Convolution.");
  PE(grad_u(sum(conv<Scal>(u, -4, 1, 1, 1, 1) * matr)));
  using W = std::array<Scal, 9>;
  PE(grad_u(sum(conv<Scal>(u, W{0, 1, 0, 1, -4, 1, 0, 1, 0}) * matr)));
  PE(grad_u(sum(conv<Scal>(u, W{1, 0, 1, 0, -4, 0, 1, 0, 1}) * matr)));
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
  TestMatrix();
  TestMultigrid();
}
