#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "matrix_cl.h"
#include "opencl.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << (a) << std::endl;
#define PEN(a) std::cout << #a << ":\n" << (a) << '\n' << std::endl;

using CL = OpenCL;

static CL Init() {
  CL::Config config;
  config.platform = 0;
  config.verbose = 0;
  config.global_size = {32, 32};
  CL cl(config);
  std::cout << "Device: " << cl.device_info_.name << std::endl;
  std::cout << "Local size: " << cl.local_size_[0] << "," << cl.local_size_[1]
            << std::endl;
  return cl;
}

template <class Scal = double>
static void TestBasic(CL& cl) {
  std::cout << '\n' << __func__ << std::endl;
  auto& queue = cl.queue_;
  auto& context = cl.context_;

  CL::MirroredBuffer<Scal> u;
  CL::MirroredBuffer<Scal> v;
  u.Create(context, cl.ngroups_, CL_MEM_WRITE_ONLY);
  v.Create(context, cl.ngroups_, CL_MEM_WRITE_ONLY);
  for (size_t i = 0; i < u.size(); ++i) {
    u[i] = 3;
    v[i] = 10;
  }
  u.EnqueueWrite(queue);
  v.EnqueueWrite(queue);
  queue.Finish();

  PE(cl.Sum(u) / u.size());
  PE(cl.Dot(u, v));
}

template <class Scal = double>
static void TestMatrix(CL& cl) {
  std::cout << '\n' << __func__ << std::endl;
  const size_t nrow = cl.global_size_[0];
  const size_t ncol = cl.global_size_[1];
  MatrixCL<Scal> u(nrow, ncol, cl);
  MatrixCL<Scal> v(nrow, ncol, cl);
  MatrixCL<Scal> iota(Matrix<Scal>::iota(nrow, ncol), cl);
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
  PE(iota(2, 2));
  PE(iota(2, 2) = 17);
  PE(iota(2, 2));
}

int main() {
  auto cl = Init();
  TestBasic(cl);
  TestMatrix(cl);
}
