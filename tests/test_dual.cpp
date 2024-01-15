#include <cmath>
#include <iostream>

#include "dual.h"
#include "matrix.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << (a) << std::endl;
#define PEN(a) std::cout << #a << ":\n" << (a) << '\n' << std::endl;

template <class T>
static T abs(T x) {
  return x >= T{} ? x : -x;
}

// Using sin(3 * x) = 3 sin(x) - 4 * sin(x)^3 and sin(x) \approx x
template <class T>
static T approx_sin(T x) {
  if (abs(x) < T(1e-5))
    return x;
  else {
    auto z = approx_sin(-x / 3);
    return 4 * z * z * z - 3 * z;
  }
}

template <class T = double>
static void TestDual() {
  std::cout << '\n' << __func__ << std::endl;
  auto a = SeedDual<double>(1);
  auto b = SeedDual<double>(2);
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

  auto f_tanh = [](auto x, auto y) {
    using std::tanh;
    return tanh(x - 0.25 * y);
  };
  PE(f_tanh(a, b));

  auto f_if = [](auto x, auto y) {
    if (x > y) {
      return x + y;
    }
    return x * y;
  };
  PE(f_if(a, b));
  PE(abs(b));
  const auto prec = std::cout.precision();
  std::cout.precision(14);
  PE(sin(SeedDual(1.23)));
  PE(approx_sin(SeedDual(1.23)));
  std::cout.precision(prec);
}

template <class T = double>
static void TestConfusion() {
  // Siskind JM, Pearlmutter BA.
  // Perturbation confusion and referential transparency.
  // https://www.bcl.hamilton.ie/~barak/papers/ifl2005.pdf
  std::cout << '\n' << __func__ << std::endl;

  auto Dx = [](auto fx, auto c) {  //
    return [fx, c]() { return fx(SeedDual<T>(c)).grad(); };
  };
  auto Dy_tagged = [](auto fxy, auto c) {  //
    return [fxy, c](auto x) {
      using U = Dual<T>;
      return fxy(x, SeedDual<U>(U(c))).grad();
    };
  };
  auto Dy_naive = [](auto fxy, auto c) {  //
    return [fxy, c](auto x) { return fxy(x, SeedDual<T>(c)).grad(); };
  };
  auto deriv = [&](auto Dy, T x0, T y0) {
    auto fsum = [](auto x, auto y) { return x + y; };
    auto fmul = [Dy, fsum, y0](auto x) { return x * Dy(fsum, y0)(x); };
    auto f = Dx(fmul, x0);
    return f();
  };
  PE(deriv(Dy_naive, 1, 1));   // Wrong.
  PE(deriv(Dy_tagged, 1, 1));  // Correct.
}

template <class F>
auto Grad(F func) -> auto {
  return [func](auto x) {  //
    return func(RaiseDual(x)).grad();
  };
}

template <class T = double>
static void TestNested() {
  std::cout << '\n' << __func__ << std::endl;
  auto f = [](auto x) { return pow(x, 3); };
  auto fx = Grad(f);
  auto fxx = Grad(fx);
  auto fxxx = Grad(fxx);
  T x = 0;
  PE(x);
  PE(f(x));
  PE(fx(x));
  PE(fxx(x));
  PE(fxxx(x));
}

template <class T = double>
static void TestDualMatrix() {
  std::cout << '\n' << __func__ << std::endl;
  using D = Dual<T>;
  auto x = SeedDual<T>(1);
  auto zeros = Matrix<D>::zeros(3, 3);
  auto ones = Matrix<D>::ones(3, 3);
  auto eye = Matrix<D>::eye(3, 3);
  PE((ones + x).sum().grad());
  PE((ones.sum() + x).grad());
  PE(((ones * x) * (ones * x))(0, 0).grad());
  PE(((ones * x).matmul(ones * x))(0, 0).grad());
}

int main() {
  TestDual();
  TestNested();
  TestConfusion();
  TestDualMatrix();
}
