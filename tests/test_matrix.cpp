#include <iomanip>
#include <iostream>
#include <sstream>

#include "matrix.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << (a) << std::endl;
#define PEN(a) std::cout << #a << ":\n" << (a) << '\n' << std::endl;

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
  auto matr = M::iota(6);
  auto Str = [](auto m) { return MatrixToStr(m, 4); };
  PEN(Str(matr));
  PEN(Str(matr.restrict()));
  PEN(Str(matr.restrict().interpolate()));
  PEN(Str(M::iota(1, 3).interpolate()));
  PEN(Str(matr.restrict_adjoint()));
  PEN(Str(M::iota(8).interpolate_adjoint()));
  auto ufine = M::iota(8);
  auto u = M::iota(4);
  PE(dot(u, ufine.restrict()));
  PE(dot(ufine, u.restrict_adjoint()));
  PE(dot(ufine, u.interpolate()));
  PE(dot(u, ufine.interpolate_adjoint()));
}

template <class T = double>
static void TestConv() {
  std::cout << '\n' << __func__ << std::endl;
  auto u = Matrix<T>::iota(4);
  auto Str = [](auto m) { return MatrixToStr(m); };
  PEN(Str(u));
  PE(rms(u.conv(1, 0, 0, 0, 0) - u));
  PE(rms(u.conv(0, 1, 0, 0, 0) - u.roll(1, 0)));
  PE(rms(u.conv(0, 0, 1, 0, 0) - u.roll(-1, 0)));
  PE(rms(u.conv(0, 0, 0, 1, 0) - u.roll(0, 1)));
  PE(rms(u.conv(0, 0, 0, 0, 1) - u.roll(0, -1)));
  PEN(Str(u.conv(-4, 1, 1, 1, 1)));
}

int main() {
  TestMatrix();
  TestRoll();
  TestMultigrid();
  TestConv();
}
