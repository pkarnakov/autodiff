#pragma once

#include <cmath>
#include <functional>
#include <vector>

#include "macros.h"
#include "matrix.h"
#include "reverse.h"

namespace optimizer {

template <class Scal, class M = Matrix<Scal>>
class Adam {
 public:
  using Matrix = M;
  struct Config {
    int epochs = 0;
    Scal lr = 1e-2;
    Scal beta1 = 0.9;
    Scal beta2 = 0.999;
    Scal epsilon = 1e-7;
  };
  static void Run(const Config& config, const std::vector<Matrix*>& vars,
                  const std::vector<const Matrix*>& grads,
                  std::function<void()> update_grads,
                  std::function<void(int)> callback) {
    fassert_equal(vars.size(), grads.size());
    std::vector<Matrix> mm;
    std::vector<Matrix> vv;
    for (auto* x : vars) {
      mm.emplace_back(Matrix::zeros_like(*x));
      vv.emplace_back(Matrix::zeros_like(*x));
    }

    callback(0);
    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
      update_grads();
      using std::pow;
      using std::sqrt;
      const Scal beta1_power = pow(config.beta1, epoch);
      const Scal beta2_power = pow(config.beta2, epoch);
      const Scal alpha = config.lr * sqrt(1 - beta2_power) / (1 - beta1_power);
      for (size_t i = 0; i < vars.size(); ++i) {
        auto& m = mm[i];
        auto& v = vv[i];
        const auto& g = *grads[i];
        auto& x = *vars[i];
        m += (g - m) * (1 - config.beta1);
        v += (sqr(g) - v) * (1 - config.beta2);
        x -= (m * alpha) / (sqrt(v) + config.epsilon);
      }
      callback(epoch);
    }
  }
};

}  // namespace optimizer
