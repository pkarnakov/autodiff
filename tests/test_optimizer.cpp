#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "matrix.h"
#include "optimizer.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << (a) << std::endl;
#define PEN(a) std::cout << #a << ":\n" << (a) << '\n' << std::endl;

template <class Scal = double>
static void TestAdam() {
  std::cout << '\n' << __func__ << std::endl;

  auto x = Matrix<Scal>::zeros(3) + 1;
  auto y = Matrix<Scal>::zeros(3) + 2;
  auto grad_x = Matrix<Scal>::zeros_like(x);
  auto grad_y = Matrix<Scal>::zeros_like(y);

  Scal loss = 0;

  auto update_grads = [&]() {
    loss = sum(sqr(x)) + sum(sqr(y));
    grad_x = 2 * x;
    grad_y = 2 * y;
  };
  auto callback = [&](int epoch) {
    if (epoch % 100 == 0) {
      printf("epoch=%4d loss=%.6e\n", epoch, loss);
    }
  };

  update_grads();
  PE(x);
  PE(y);
  PE(grad_x);
  PE(grad_y);

  using Adam = optimizer::Adam<Scal>;
  typename Adam::Config config;
  config.epochs = 1000;
  Adam().Run(config, {&x, &y}, {&grad_x, &grad_y}, update_grads, callback);
}

int main() {
  TestAdam();
}
