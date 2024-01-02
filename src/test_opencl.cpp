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

  using CL = OpenCL<Scal>;
  typename CL::Config config;
  config.platform = 0;
  config.verbose = 0;
  config.global_size = {32, 32};
  CL cl(config);
  auto& queue = cl.queue_;
  auto& context = cl.context_;
  auto& device = cl.device_;

  std::cout << "Device: " << cl.device_info_.name << std::endl;

  typename CL::Program program;
  typename CL::Kernel kernel;
  program.CreateFromString(kKernelSource, context, device);
  kernel.Create(program, "field_sum");
}

int main() {
  TestOpenCL();
}
