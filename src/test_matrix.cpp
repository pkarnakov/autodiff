#include <iomanip>
#include <iostream>
#include <sstream>

#include "matrix.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << '\n' << std::endl;

template <class T>
std::string Str(const Matrix<T>& matr) {
  std::stringstream out;
  for (size_t i = 0; i < matr.nrow(); ++i) {
    for (size_t j = 0; j < matr.ncol(); ++j) {
      out << std::setw(3) << matr(i, j);
      if (j + 1 < matr.ncol()) {
        out << ' ';
      }
    }
    if (i + 1 < matr.nrow()) {
      out << '\n';
    }
  }
  return out.str();
};

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
  Matrix<T> matr(5);
  for (size_t i = 0; i < matr.size(); ++i) {
    matr[i] = i;
  }
  PEN(Str(matr));
  PEN(Str(matr.roll(0, 1)));
  PEN(Str(matr.roll(1, 0)));
  PEN(Str(matr.roll(1, 1)));
  PEN(Str(matr.roll(0, -1)));
  PEN(Str(matr.roll(-1, 0)));
  PEN(Str(matr.roll(-1, -1)));
}

int main() {
  TestMatrix();
  TestRoll();
}
