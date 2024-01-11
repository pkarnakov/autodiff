#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "matrix_cl.h"
#include "opencl.h"
#include "reverse.h"
#include "reverse_cl.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << (a) << std::endl;
#define PEN(a) std::cout << #a << ":\n" << (a) << '\n' << std::endl;

using CL = OpenCL;

static CL Init(bool verbose = false) {
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
static void TestBasic(CL& cl, size_t nrow) {
  std::cout << '\n' << __func__ << std::endl;
  auto& queue = cl.queue_;
  auto& context = cl.context_;

  const CL::MSize nw{nrow, nrow};
  CL::MirroredBuffer<Scal> u;
  CL::MirroredBuffer<Scal> v;
  u.Create(context, nw[0] * nw[1], CL_MEM_WRITE_ONLY);
  v.Create(context, nw[0] * nw[1], CL_MEM_WRITE_ONLY);
  for (size_t i = 0; i < u.size(); ++i) {
    u[i] = 3;
    v[i] = 10;
  }
  u.EnqueueWrite(queue);
  v.EnqueueWrite(queue);

  PE(cl.Sum(nw, u) / u.size());
  PE(cl.Dot(nw, u, v) / u.size());
}

template <class Scal = double>
static void TestMatrix(CL& cl, size_t nrow) {
  std::cout << '\n' << __func__ << std::endl;
  auto Str = [](auto m) { return MatrixToStr(m, 3, 3); };
  const size_t ncol = nrow;
  MatrixCL<Scal> u(nrow, ncol, cl);
  MatrixCL<Scal> v(nrow, ncol, cl);
  const MatrixCL<Scal> iota(Matrix<Scal>::iota(nrow, ncol), cl);
  u.fill(3);
  v.fill(10);
  PE(u.mean());
  PE(u.max());
  PE(u.min());
  PE(u.dot(v) / u.size());
  PE((u + v).mean());
  PE((u - v).mean());
  PE((u * v).mean());
  PE((u / v).mean());
  PE((u + 2).mean());
  PE((2 + u).mean());
  PE((u - 2).mean());
  PE((2 - u).mean());
  PE((u * 2).mean());
  PE((2 * u).mean());
  PE((sin(u)).mean());
  PE((cos(u)).mean());
  PE((exp(u)).mean());
  PE((log(u)).mean());
  PE(Matrix<Scal>(u).mean());
  PE(MatrixCL<Scal>(Matrix<Scal>(u) + 1, cl).mean());
  PE(u(2, 2));
  PE(u(2, 2) = 17);
  PE(u(2, 2));
  PE((u = v, u.mean()));
  PE((u += v, u.mean()));
  MatrixCL<Scal> w;
  w = u;
  PE((w = u, w.mean()));
  PEN(Str(iota.roll(0, 0)));
  PEN(Str(iota.roll(1, 0)));
  PEN(Str(iota.roll(0, 1)));
  PEN(Str(iota.roll(1, 1)));
  PEN(Str(iota.roll(-1, 0)));
  PEN(Str(iota.roll(0, -1)));
  PEN(Str(iota.roll(-1, -1)));

  u = iota;
  PEN(Str(u));
  PE(rms(u.conv(1, 0, 0, 0, 0) - u));
  PE(rms(u.conv(0, 1, 0, 0, 0) - u.roll(1, 0)));
  PE(rms(u.conv(0, 0, 1, 0, 0) - u.roll(-1, 0)));
  PE(rms(u.conv(0, 0, 0, 1, 0) - u.roll(0, 1)));
  PE(rms(u.conv(0, 0, 0, 0, 1) - u.roll(0, -1)));
  PEN(Str(u.conv(-4, 1, 1, 1, 1)));
}

template <class Scal = double>
static void TestMultigrid(CL& cl, size_t nrow) {
  std::cout << '\n' << __func__ << std::endl;
  auto Str = [](auto m) { return MatrixToStr(m, 3, 3); };
  const size_t ncol = nrow;
  MatrixCL<Scal> u(Matrix<Scal>::iota(nrow, ncol), cl);
  PEN(Str(u));
  PEN(Str(u.restrict()));
  PEN(Str(u.restrict().restrict_adjoint()));
  PEN(Str(u.restrict().interpolate()));
  PEN(Str(u.interpolate_adjoint()));
}

struct Extra : public BaseExtra {
  Extra(std::ostream& out) : dot(out) {}
  template <class Node>
  void Visit(Node* node) {
    dot.Write(node);
  }
  DotWriter dot;
};

template <class Scal = double>
static void TestReverse(CL& cl, size_t nrow) {
  std::cout << '\n' << __func__ << std::endl;
  const auto pi = M_PI;
  auto Str = [](auto m) { return MatrixToStr(m, 3, 1); };

  const size_t ncol = nrow;
  const auto eye = Matrix<Scal>::eye(nrow, ncol);
  auto matr_x = std::make_unique<MatrixCL<Scal>>(eye * (pi / 8), cl);
  auto matr_y = std::make_unique<MatrixCL<Scal>>(eye * (pi / 8), cl);
  Var<MatrixCL<Scal>> var_x(std::move(matr_x), "x");
  Var<MatrixCL<Scal>> var_y(std::move(matr_y), "y");
  PE(var_x);
  PE(var_y);
  auto x = MakeTracer<Extra>(var_x);
  auto y = MakeTracer<Extra>(var_y);

  auto eval = [&](auto& e, std::string path) {
    const auto order = e.GetFowardOrder();
    e.UpdateGrad(order);
    PE(e.value());
    PEN(Str(x.grad()));
    PEN(Str(y.grad()));
    std::cout << path << std::endl;
    std::ofstream fout(path);
    Extra extra(fout);
    Traverse(order, extra);
    ClearVisited(order);
  };
  auto e1 = sum(sin(x) * cos(y) + cos(x) * sin(y));
  eval(e1, "reverse_cl_1.gv");
  auto e2 = sum(sin(x + y));
  eval(e2, "reverse_cl_2.gv");
}

int main() {
  auto cl = Init(true);
  TestBasic(cl, 16);
  TestReverse(cl, 16);
  TestMatrix(cl, 4);
  TestMultigrid(cl, 8);
}
