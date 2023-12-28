#include <iostream>

#include "matrix.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << '\n' << std::endl;

template <class T = double>
static void TestMatrix() {
  std::cout << '\n' << __func__ << std::endl;
  auto zeros = Matrix<T>::zeros(3, 3);
  auto izeros = Matrix<int>::zeros(3, 3);
  auto eye = Matrix<T>::eye(3, 3);
  auto ieye = Matrix<int>::eye(3, 3);
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

int main() {
  TestMatrix();
}
