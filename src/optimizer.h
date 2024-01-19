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
  struct State {
    int epoch = 0;
    std::vector<Matrix> mm;
    std::vector<Matrix> vv;
    std::vector<Scal> mm_scal;
    std::vector<Scal> vv_scal;
  };

  Adam() = default;
  void Run(const Config& config, const std::vector<Matrix*>& vars,
           const std::vector<const Matrix*>& grads,
           const std::vector<Scal*>& vars_scal,
           const std::vector<const Scal*>& grads_scal,
           const std::function<void()>& update_grads,
           const std::function<void(int)>& callback) {
    fassert_equal(vars.size(), grads.size());
    auto& mm = state_.mm;
    auto& vv = state_.vv;
    auto& mm_scal = state_.mm_scal;
    auto& vv_scal = state_.vv_scal;
    // Initialize mm and vv on first call.
    if (mm.empty() && vv.empty() && mm_scal.empty() && vv_scal.empty()) {
      for (auto* x : vars) {
        mm.emplace_back(Matrix::zeros_like(*x));
        vv.emplace_back(Matrix::zeros_like(*x));
      }
      mm_scal.resize(vars_scal.size(), Scal(0));
      vv_scal.resize(vars_scal.size(), Scal(0));
    } else {
      fassert_equal(mm.size(), vars.size());
      fassert_equal(vv.size(), vars.size());
      fassert_equal(vv_scal.size(), vars_scal.size());
      fassert_equal(mm_scal.size(), vars_scal.size());
    }

    const int epoch_start = state_.epoch;
    if (epoch_start == 0) {
      callback(epoch_start);
    }
    for (int epoch = epoch_start + 1; epoch <= epoch_start + config.epochs;
         ++epoch) {
      update_grads();
      using std::pow;
      using std::sqrt;
      const Scal beta1_power = pow(config.beta1, epoch);
      const Scal beta2_power = pow(config.beta2, epoch);
      const Scal alpha = config.lr * sqrt(1 - beta2_power) / (1 - beta1_power);
      auto step = [&](auto& m, auto& v, auto& x, const auto& g) {
        m += (g - m) * (1 - config.beta1);
        v += (sqr(g) - v) * (1 - config.beta2);
        x -= (m * alpha) / (sqrt(v) + config.epsilon);
      };
      for (size_t i = 0; i < vars.size(); ++i) {
        step(mm[i], vv[i], *vars[i], *grads[i]);
      }
      for (size_t i = 0; i < vars_scal.size(); ++i) {
        step(mm_scal[i], vv_scal[i], *vars_scal[i], *grads_scal[i]);
      }
      callback(epoch);
    }
    state_.epoch = epoch_start + config.epochs;
  }

  void Run(const Config& config, const std::vector<Matrix*>& vars,
           const std::vector<const Matrix*>& grads,
           const std::function<void()>& update_grads,
           const std::function<void(int)>& callback) {
    Run(config, vars, grads, {}, {}, update_grads, callback);
  }
  State& state() {
    return state_;
  }

 private:
  State state_;
};

}  // namespace optimizer
