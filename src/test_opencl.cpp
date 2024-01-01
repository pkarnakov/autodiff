#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

// Print and evaluate.
#define PE(a) std::cout << #a << ": " << a << std::endl;
#define PEN(a) std::cout << #a << ":\n" << a << '\n' << std::endl;

const char* kKernels =
#include "kernels.inc"
    ;

template <class Scal = double>
static void TestOpenCL() {
  std::cout << '\n' << __func__ << std::endl;
}

int main() {
  TestOpenCL();
}
