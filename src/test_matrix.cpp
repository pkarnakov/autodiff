#include <iomanip>
#include <iostream>
#include <sstream>

#include "matrix.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << '\n' << std::endl;

template <class T = double>
static void TestMatrix() {
  std::cout << '\n' << __func__ << std::endl;
  auto zeros = Matrix<T>::zeros(3);
  auto izeros = Matrix<int>::zeros(3);
  auto eye = Matrix<T>::eye(3);
  auto ieye = Matrix<int>::eye(3);
  PEN(eye + ieye);
  PEN(-eye - ieye);
  PEN(8 / (2 + 2 * eye * 2 + 2));
  PEN(1 - eye - 1);
  PEN(ieye.apply([](auto a) { return a / 2 + 0.5; }));
  PEN(eye * (eye + 1));
  PEN(eye.matmul(eye + 1));
  PEN(eye.vstack(eye).transpose() + eye.hstack(eye));
  PEN(eye.sum());
  PEN(eye.mean());
}

template <class T = double>
static void TestRoll() {
  std::cout << '\n' << __func__ << std::endl;
  auto matr = Matrix<T>::iota(5);
  auto Str = [](auto m) { return MatrixToStr(m); };
  PEN(Str(matr));
  PEN(Str(matr.roll(0, 1)));
  PEN(Str(matr.roll(1, 0)));
  PEN(Str(matr.roll(1, 1)));
  PEN(Str(matr.roll(0, -1)));
  PEN(Str(matr.roll(-1, 0)));
  PEN(Str(matr.roll(-1, -1)));
}

template <class T = double>
static void TestMultigrid() {
  std::cout << '\n' << __func__ << std::endl;
  using M = Matrix<T>;
  auto matr = M::iota(8);
  auto Str = [](auto m) { return MatrixToStr(m, 4); };
  PEN(Str(matr));
  PEN(Str(matr.restrict()));
  PEN(Str(matr.restrict().interpolate()));
  PEN(Str(M::iota(1, 4).interpolate()));
  PEN(Str(M::iota(4, 1).interpolate()));
}

int main() {
  TestMatrix();
  TestRoll();
  TestMultigrid();
}
