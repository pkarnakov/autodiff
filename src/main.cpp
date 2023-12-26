#include <cmath>
#include <iostream>
#include <vector>

#include "dual.h"
#include "matrix.h"
#include "tape.h"

template <class T>
T tanh(T x) {
  using std::exp;
  return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << std::endl;

void TestDual() {
  std::cout << '\n' << __func__ << std::endl;
  auto a = Trace<double>(1);
  auto b = Trace<double>(2);
  PE(a);
  PE(b);
  PE(a + 1);
  PE(a + 1);
  PE(a - 1);
  PE(a * 2);
  PE(a / 2);
  PE(1 + a);
  PE(1 - a);
  PE(2 * a);
  PE(2 / a);
  PE(a + b);
  PE(a - b);
  PE(a * b);
  PE(a / b);
  PE(sin(a));
  PE(cos(a));
  PE(exp(a));
  PE(log(a));
  PE(pow(a, 4.5));
  PE(tanh(a));

  auto f = [](auto x, auto y) {
    using std::tanh;
    return tanh(x - 0.25 * y);
  };
  PE(f(a, b));
}

template <class F>
auto Grad(F func) -> auto {
  return [func](auto x) {  //
    return func(Trace(x)).grad();
  };
}

void TestNested() {
  std::cout << '\n' << __func__ << std::endl;
  auto f = [](auto x) { return pow(x, 3); };
  auto fx = Grad(f);
  auto fxx = Grad(fx);
  auto fxxx = Grad(fxx);
  double x = 0;
  PE(x);
  PE(f(x));
  PE(fx(x));
  PE(fxx(x));
  PE(fxxx(x));
}

void TestCond() {
  std::cout << '\n' << __func__ << std::endl;
  auto f = [](auto a, auto b) {
    if (a > b) {
      return a + b;
    }
    return a * b;
  };
  auto x = Trace(0);
  auto a = x + 3;
  auto b = x + 2;
  std::cout << f(a, b) << std::endl;
}

void TestMatrix() {
  std::cout << '\n' << __func__ << std::endl;
  auto x = Trace<double>(1);
  auto a = Matrix<double>::zeros(3, 3);
  auto b = Matrix<double>::eye(3, 3);
  PEN(a + x);
  PEN(b * x + 1);
  PEN((b * x).sum().grad());
}

void TestDualMatrix() {
  std::cout << '\n' << __func__ << std::endl;
  auto eye = Matrix<double>::eye(3, 3);
  auto zeros = Matrix<double>::zeros(3, 3);
  auto dual = Dual<double, Matrix<double>>(eye, eye);
  PEN(dual);
  PEN(dual + eye);
}

int main() {
  TestDual();
  TestNested();
  TestCond();
  TestMatrix();
  TestDualMatrix();
}
