#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "opencl.h"

const char* kKernelSource =
#include "kernels.inc"
    ;

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << '\n' << std::endl;

template <class Scal = double>
static void TestOpenCL() {
  std::cout << '\n' << __func__ << std::endl;

  using CL = OpenCL;
  CL::Config config;
  config.platform = 0;
  config.verbose = 0;
  config.global_size = {32, 32};
  CL cl(config);
  auto& queue = cl.queue_;
  auto& context = cl.context_;
  auto& device = cl.device_;

  std::cout << "Device: " << cl.device_info_.name << std::endl;
  std::cout << "Local size: " << cl.local_size_[0] << "," << cl.local_size_[1]
            << std::endl;

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

int main() {
  TestOpenCL();
}
