#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "matrix.h"
#include "reverse.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << std::endl;

struct Extra : public BaseExtra {
  Extra(std::ostream& out) : dot(out) {}
  template <class N>
  void TraversePre(N* node) {
    dot.Write(node);
  }
  DotWriter dot;
};

struct Config {
  int Nx = 64;
  int epochs = 1000;
  int frames = 10;
  double eta = 1e-2;
};

template <class Scal = double>
static void RunPoisson(Config config) {
  // Writes graph to DOT file.
  auto dump_graph = [](auto& e, std::string path) {
    std::ofstream fout(path);
    Extra extra(fout);
    e.TraversePre(extra);
  };
  // Writes field to DAT file.
  auto dump_field = [](auto u, auto path) {
    std::ofstream fout(path);
    fout << MatrixToStr(u) << std::endl;
  };

  const size_t Nx = config.Nx;

  // Reference solution.
  auto uref = Matrix<Scal>::zeros(Nx);
  for (size_t i = 0; i < Nx; ++i) {
    for (size_t j = 0; j < Nx; ++j) {
      const Scal x = (i + 0.5) / Nx;
      const Scal y = (j + 0.5) / Nx;
      uref(i, j) = (sqr(x - 0.5) + sqr(y - 0.5) < sqr(0.25) ? 1 : 0);
    }
  }
  auto eval_lapl = [](auto u) {
    return (roll(u, 1, 0) + roll(u, -1, 0)) +  //
           (roll(u, 0, 1) + roll(u, 0, -1)) - 4 * u;
  };

  auto rhs = eval_lapl(uref);
  Var var_u(Matrix<Scal>::zeros_like(rhs), "u");
  auto u = MakeTracer<Extra>(var_u);
  auto lapl = eval_lapl(u);
  auto loss = sum(sqr(lapl - rhs));

  dump_graph(loss, "poisson.gv");

  auto grad = [&u](auto& e) {
    e.UpdateGrad();
    return u.grad();
  };

  dump_field(uref, "uref.dat");

  auto time_prev = std::chrono::steady_clock::now();
  const int dump_every = std::max(1, config.epochs / config.frames);
  int frame = 0;
  for (int epoch = 0; epoch <= config.epochs; ++epoch) {
    if (epoch % dump_every == 0) {
      auto time_curr = std::chrono::steady_clock::now();
      auto delta = time_curr - time_prev;
      time_prev = time_curr;
      auto msdur = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
      auto ms = msdur.count();
      double throughput = ms > 0 ? 1e-3 * rhs.size() * dump_every / ms : 0;
      printf("epoch=%5d, loss=%8.6e, throughput=%.3fM cells/s\n", epoch,
             loss.value(), throughput);
      std::string path = [&]() {
        std::stringstream buf;
        buf << "u_" << std::setfill('0') << std::setw(5) << frame << ".dat";
        return buf.str();
      }();
      dump_field(var_u.value(), path);
      ++frame;
    }
    if (epoch == config.epochs) {
      break;
    }
    var_u.value() -= config.eta * grad(loss);
  }
}

int main() {
  Config config;
  RunPoisson(config);
}
