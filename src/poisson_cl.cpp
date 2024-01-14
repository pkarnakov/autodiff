#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "matrix_cl.h"
#include "opencl.h"
#include "optimizer.h"
#include "reverse_cl.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << (a) << std::endl;
#define PEN(a) std::cout << #a << ":\n" << (a) << std::endl;

struct Extra : public BaseExtra {
  Extra(std::ostream& out) : dot(out) {}
  template <class N>
  void Visit(N* node) {
    dot.Write(node);
  }
  DotWriter dot;
};

struct Config {
  int Nx = 64;
  int epochs = 1000;
  int frames = 10;
  int max_nlvl = 4;
  double lr = 1e-3;
  double uref_k = 2;
};

using CL = OpenCL;

static CL Init(size_t nx, bool verbose = false) {
  CL::Config config;
  config.platform = 0;
  config.verbose = 0;
  CL cl(config);
  if (verbose) {
    std::cerr << "Device: " << cl.device_info_.name << std::endl;
    std::cerr << "Local size: " << cl.local_size_[0] << "," << cl.local_size_[1]
              << std::endl;
  }
  return cl;
}

template <class Scal = double>
static void RunPoisson(Config config) {
  // Writes graph to DOT file.
  auto dump_graph = [](const auto& order, std::string path) {
    std::ofstream fout(path);
    Extra extra(fout);
    Traverse(order, extra);
  };
  // Writes field to DAT file.
  auto dump_field = [](auto u, auto path) {
    std::ofstream fout(path);
    fout << MatrixToStr(Matrix<Scal>(u).transpose()) << std::endl;
  };

  const size_t Nx = config.Nx;
  const Scal hx = 1. / Nx;
  auto cl = Init(Nx, true);

  // Reference solution.
  auto uref = MatrixCL<Scal>::zeros(Nx, Nx, cl);
  for (size_t i = 0; i < Nx; ++i) {
    for (size_t j = 0; j < Nx; ++j) {
      using std::sin;
      const Scal x = (i + 0.5) / Nx;
      const Scal y = (j + 0.5) / Nx;
      const Scal pi = M_PI;

      const Scal k = config.uref_k;
      uref(i, j) = sin(pi * sqr(k * x)) * sin(pi * y);
    }
  }
  auto eval_lapl = [hx](auto& u) {  //
    return conv(u, -4, 1, 1, 1, 1) / sqr(hx);
  };

  using M = MatrixCL<Scal>;
  auto rhs = eval_lapl(uref);
  std::vector<std::unique_ptr<Var<M>>> var_uu;
  std::vector<Tracer<M, Extra>> uu;
  int nx = Nx;
  // Create variables for multigrid levels.
  for (int i = 0; i < config.max_nlvl; ++i) {
    const auto name = "u" + std::to_string(i + 1);
    var_uu.emplace_back(std::make_unique<Var<M>>(M::zeros(nx, nx, cl), name));
    uu.emplace_back(*var_uu.back());
    nx /= 2;
    if (nx <= 4) {
      break;
    }
  }
  std::cout << "multigrid levels: ";
  for (auto& var_u : var_uu) {
    auto& m = var_u->value();
    std::cout << "(" << m.nrow() << "," << m.ncol() << ") ";
  }
  std::cout << std::endl;
  auto u = uu.back();
  for (size_t i = uu.size() - 1; i > 0;) {
    --i;
    u = interpolate(u) + uu[i];
  }
  auto loss = mean(sqr(eval_lapl(u) - rhs));
  const NodeOrder<Extra> order = loss.GetFowardOrder();
  dump_graph(order, "poisson.gv");
  loss.UpdateValue(order);  // Required before calling the optimizer.
  dump_field(uref, "uref.dat");

  auto time_prev = std::chrono::steady_clock::now();
  const int dump_every = std::max(1, config.epochs / config.frames);
  int frame = 0;

  auto callback = [&](int epoch) {
    if (epoch % dump_every == 0) {
      auto time_curr = std::chrono::steady_clock::now();
      auto delta = time_curr - time_prev;
      time_prev = time_curr;
      auto msdur = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
      auto ms = msdur.count();
      const double throughput =
          (epoch > 0 && ms > 0 ? 1e-3 * rhs.size() * dump_every / ms : 0);
      printf(
          "epoch=%5d, loss=%8.6e, throughput=%.3fM cells/s"
          ", u:[%.3f,%.3f], \n",
          epoch, loss.value(), throughput, u.value().min(), u.value().max());
      std::string path = [&]() {
        std::stringstream buf;
        buf << "u_" << std::setfill('0') << std::setw(5) << frame << ".dat";
        return buf.str();
      }();
      dump_field(u.value(), path);
      ++frame;
    }
  };

  auto update_grads = [&]() { loss.UpdateGrad(order); };

  using Adam = optimizer::Adam<Scal, M>;
  typename Adam::Config adam_config;
  adam_config.epochs = config.epochs;
  adam_config.lr = config.lr;
  std::vector<M*> vars;
  std::vector<const M*> grads;
  for (size_t i = 0; i < var_uu.size(); ++i) {
    vars.push_back(&var_uu[i]->value());
    grads.push_back(&uu[i].grad());
  }
  Adam().Run(adam_config, vars, grads, update_grads, callback);

  auto* profiler = Profiler::GetInstance();
  for (auto& pair : profiler->GetRecords()) {
    std::cout << '\n'
              << pair.first << '\n'
              << to_string(pair.second) << std::endl;
  }
}

int main() {
  Config config;
  RunPoisson(config);
}
