#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "opencl.h"

const char* kKernelSource =
#include "kernels.inc"
    ;

#include "matrix_cl.h"

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << '\n' << std::endl;

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
  auto& device = cl.device_;

  CL::Program program;
  CL::Kernel kernel;
  program.CreateFromString(kKernelSource, context, device);

  CL::MirroredBuffer<Scal> u;
  CL::MirroredBuffer<Scal> v;
  u.Create(context, cl.ngroups_, CL_MEM_WRITE_ONLY);
  v.Create(context, cl.ngroups_, CL_MEM_WRITE_ONLY);
  for (size_t i = 0; i < u.size(); ++i) {
    u[i] = 17. / u.size();
    v[i] = 10;
  }
  u.EnqueueWrite(queue);
  v.EnqueueWrite(queue);
  queue.Finish();

  PE(cl.Sum(u));
  PE(cl.Dot(u, v));
  // kernel.Create(program, "restrict");
}

template <class Scal = double>
static void TestMatrix(CL& cl) {
  std::cout << '\n' << __func__ << std::endl;
  const size_t nrow = cl.global_size_[0];
  const size_t ncol = cl.global_size_[1];
  MatrixCL<Scal> u(nrow, ncol, cl);
  MatrixCL<Scal> v(nrow, ncol, cl);
  u.fill(17);
  v.fill(10);
  PE(u.mean());
  PE(u.max());
  PE(u.dot(v));
  PE((u + v).mean());
  PE((u - v).mean());
  PE((u * v).mean());
  PE((u / v).mean());
  PE(Matrix<Scal>(u).mean());
  PE(MatrixCL<Scal>(Matrix<Scal>(u) + 1, cl).mean());
}

int main() {
  auto cl = Init();
  TestBasic(cl);
  TestMatrix(cl);
}
