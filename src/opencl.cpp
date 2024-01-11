#include "opencl.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

static const char* kKernelSource =
#include "kernels.inc"
    ;

std::string OpenCL::GetErrorMessage(cl_int error) {
  switch (error) {
    case CL_SUCCESS:
      return "";
    case CL_INVALID_VALUE:
      return "invalid value";
    case CL_OUT_OF_HOST_MEMORY:
      return "out of host memory";
    case CL_PLATFORM_NOT_FOUND_KHR:
      return "platform not found";
    default:
      return "unknown error " + std::to_string(error);
  }
  return "";
}

auto OpenCL::Device::GetPlatformInfos() -> std::vector<PlatformInfo> {
  cl_uint nplatforms = 0;
  std::array<cl_platform_id, 10> platforms;
  cl_int error;
  error = clGetPlatformIDs(platforms.size(), platforms.data(), &nplatforms);
  fassert(error == CL_SUCCESS, GetErrorMessage(error));

  std::vector<PlatformInfo> res(nplatforms);

  for (size_t i = 0; i < nplatforms; ++i) {
    auto& info = res[i];
    info.id = platforms[i];

    std::vector<char> name(1024, '\0');
    clGetPlatformInfo(info.id, CL_PLATFORM_NAME, name.size() - 1, name.data(),
                      NULL);
    info.name = std::string(name.data());

    std::vector<char> vendor(1024, '\0');
    clGetPlatformInfo(info.id, CL_PLATFORM_VENDOR, vendor.size() - 1,
                      vendor.data(), NULL);
    info.vendor = std::string(vendor.data());
  }
  return res;
}

cl_device_id OpenCL::Device::GetDevice(cl_platform_id platform) {
  cl_device_id device;
  cl_int error;
  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (error == CL_DEVICE_NOT_FOUND) {
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  }
  fassert(error == CL_SUCCESS, GetErrorMessage(error));
  return device;
}

auto OpenCL::Device::GetDeviceInfo(cl_platform_id platform) -> DeviceInfo {
  DeviceInfo info;
  info.id = GetDevice(platform);

  std::vector<char> name(1024, '\0');
  CLCALL(clGetDeviceInfo(info.id, CL_DEVICE_NAME, name.size() - 1, name.data(),
                         NULL));
  info.name = std::string(name.data());

  std::vector<char> ext(65536, '\0');
  CLCALL(clGetDeviceInfo(info.id, CL_DEVICE_EXTENSIONS, ext.size() - 1,
                         ext.data(), NULL));
  info.extensions = std::string(ext.data());

  CLCALL(clGetDeviceInfo(info.id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof(info.max_work_size), &info.max_work_size,
                         NULL));
  info.extensions = std::string(ext.data());

  return info;
}

void OpenCL::Device::Create(size_t pid) {
  auto infos = GetPlatformInfos();
  fassert(pid < infos.size(),  //
          "Invalid platform index " + std::to_string(pid) +
              ". Maximum valid index is " + std::to_string(infos.size() - 1));
  platform = infos[pid].id;
  handle = GetDevice(platform);
}

OpenCL::Device::~Device() {
  if (handle) {
    clReleaseDevice(handle);
  }
}

void OpenCL::Context::Create(const cl_device_id* device) {
  cl_int error;
  handle = clCreateContext(NULL, 1, device, NULL, NULL, &error);
  CLCALL(error);
}

OpenCL::Context::~Context() {
  if (handle) {
    clReleaseContext(handle);
  }
}

void OpenCL::Queue::Create(cl_context context, cl_device_id device) {
  cl_int error;
  handle = clCreateCommandQueue(context, device, 0, &error);
  CLCALL(error);
}

void OpenCL::Queue::Finish() {
  CLCALL(clFinish(handle));
}

OpenCL::Queue::~Queue() {
  if (handle) {
    clReleaseCommandQueue(handle);
  }
}

void OpenCL::Program::CreateFromString(std::string source, cl_context context,
                                       cl_device_id device) {
  std::stringstream flags;
  flags << " -DScal=" << (sizeof(Scal) == 4 ? "float" : "double");
  flags << " -cl-std=CL2.0";

  int error;

  const char* source_str = source.c_str();
  size_t source_size = source.length();
  handle =
      clCreateProgramWithSource(context, 1, &source_str, &source_size, &error);
  CLCALL(error);

  error = clBuildProgram(handle, 0, NULL, flags.str().c_str(), NULL, NULL);
  if (error != CL_SUCCESS) {
    size_t logsize = 0;
    clGetProgramBuildInfo(handle, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &logsize);
    std::vector<char> log(logsize, '\0');
    clGetProgramBuildInfo(handle, device, CL_PROGRAM_BUILD_LOG, log.size(),
                          log.data(), NULL);
    std::cerr << std::string(log.data());
    CLCALL(error);
  }
}

void OpenCL::Program::CreateFromStream(std::istream& in, cl_context context,
                                       cl_device_id device) {
  std::stringstream ss;
  ss << in.rdbuf();
  CreateFromString(ss.str(), context, device);
}

void OpenCL::Program::CreateFromFile(std::string source_path,
                                     cl_context context, cl_device_id device) {
  std::ifstream fin(source_path);
  CreateFromStream(fin, context, device);
}

OpenCL::Program::~Program() {
  if (handle) {
    clReleaseProgram(handle);
  }
}

void OpenCL::Kernel::Create(cl_program program, std::string name_) {
  name = name_;
  cl_int error;
  handle = clCreateKernel(program, name.c_str(), &error);
  fassert_equal(error, CL_SUCCESS, ". Kernel '" + name + "' not found");
}

OpenCL::Kernel::~Kernel() {
  if (handle) {
    clReleaseKernel(handle);
  }
}

OpenCL::OpenCL(const Config& config) {
  device_.Create(config.platform);
  device_info_ = Device::GetDeviceInfo(device_.platform);

  fassert_equal(kDim, 2);
  // Determine the maximum allowed local size.
  size_t lsize = 1;
  while (std::pow(lsize * 2, 2) <= device_info_.max_work_size) {
    lsize *= 2;
  }
  local_size_ = {lsize, lsize};

  if (config.verbose) {
    auto pinfo = Device::GetPlatformInfos()[config.platform];
    std::cout << "Platform name: " << pinfo.name         //
              << "\nPlatform vendor: " << pinfo.vendor   //
              << "\nDevice name: " << device_info_.name  //
              << "\nDevice extensions:\n"
              << device_info_.extensions << '\n';
  }

  context_.Create(&device_.handle);
  queue_.Create(context_, device_);

  program_.CreateFromString(kKernelSource, context_, device_);
  for (std::string name : {"reduce_max",        "reduce_min",
                           "reduce_sum",        "reduce_dot",
                           "assign_fill",       "assign_add",
                           "assign_sub",        "assign_subarray",
                           "scalar_add",        "scalar_sub",
                           "scalar_sub2",       "scalar_mul",
                           "scalar_div",        "scalar_div2",
                           "field_add",         "field_sub",
                           "field_mul",         "field_div",
                           "unary_sin",         "unary_cos",
                           "unary_exp",         "unary_log",
                           "unary_sqr",         "unary_sqrt",
                           "unary_roll",        "unary_conv",
                           "field_restrict",    "field_restrict_adjoint",
                           "field_interpolate", "field_interpolate_adjoint"}) {
    kernels_[name].Create(program_, name);
  }
}

auto OpenCL::Max(MSize nw, cl_mem u) -> Scal {
  const size_t ngroups = GetNumGroups(nw);
  MirroredBuffer<Scal> d_buf_reduce(context_, ngroups, CL_MEM_WRITE_ONLY);
  Launch("reduce_max", nw, u, d_buf_reduce);
  d_buf_reduce.EnqueueRead(queue_);
  Scal res = -std::numeric_limits<Scal>::max();
  for (size_t i = 0; i < ngroups; ++i) {
    res = std::max(res, d_buf_reduce[i]);
  }
  return res;
}

auto OpenCL::Min(MSize nw, cl_mem u) -> Scal {
  const size_t ngroups = GetNumGroups(nw);
  MirroredBuffer<Scal> d_buf_reduce(context_, ngroups, CL_MEM_WRITE_ONLY);
  Launch("reduce_min", nw, u, d_buf_reduce);
  d_buf_reduce.EnqueueRead(queue_);
  Scal res = std::numeric_limits<Scal>::max();
  for (size_t i = 0; i < ngroups; ++i) {
    res = std::min(res, d_buf_reduce[i]);
  }
  return res;
}

auto OpenCL::Sum(MSize nw, cl_mem u) -> Scal {
  const size_t ngroups = GetNumGroups(nw);
  MirroredBuffer<Scal> d_buf_reduce(context_, ngroups, CL_MEM_WRITE_ONLY);
  Launch("reduce_sum", nw, u, d_buf_reduce);
  d_buf_reduce.EnqueueRead(queue_);
  Scal res = 0;
  for (size_t i = 0; i < ngroups; ++i) {
    res += d_buf_reduce[i];
  }
  return res;
}

auto OpenCL::Dot(MSize nw, cl_mem u, cl_mem v) -> Scal {
  const size_t ngroups = 1;
  MirroredBuffer<Scal> d_buf_reduce(context_, ngroups, CL_MEM_WRITE_ONLY);
  Launch("reduce_dot", nw, u, v, d_buf_reduce);
  d_buf_reduce.EnqueueRead(queue_);
  Scal res = 0;
  for (size_t i = 0; i < ngroups; ++i) {
    res += d_buf_reduce[i];
  }
  return res;
}

template <class T>
auto OpenCL::ReadAt(MSize nw, cl_mem u, MInt iw) -> T {
  T res;
  CLCALL(clEnqueueReadBuffer(queue_, u, CL_TRUE,
                             sizeof(T) * (iw[1] * nw[0] + iw[0]), sizeof(T),
                             &res, 0, NULL, NULL));
  return res;
}

template <class T>
void OpenCL::WriteAt(MSize nw, cl_mem u, MInt iw, T value) {
  CLCALL(clEnqueueWriteBuffer(queue_, u, CL_TRUE,
                              sizeof(T) * (iw[1] * nw[0] + iw[0]), sizeof(T),
                              &value, 0, NULL, NULL));
}

void OpenCL::Fill(MSize nw, cl_mem u, Scal value) {
  Launch("assign_fill", nw, u, value);
}

void OpenCL::AssignAdd(MSize nw, cl_mem u, cl_mem v) {
  Launch("assign_add", nw, u, v);
}

void OpenCL::AssignSub(MSize nw, cl_mem u, cl_mem v) {
  Launch("assign_sub", nw, u, v);
}

void OpenCL::AssignSubarray(MInt nw_u, MInt nw_v, cl_mem u, cl_mem v, MInt iu,
                            MInt iv, MSize icnt) {
  Launch("assign_subarray", icnt, u, v, nw_u[0], nw_u[1], nw_v[0], nw_v[1],
         iu[0], iu[1], iv[0], iv[1]);
}

void OpenCL::Add(MSize nw, cl_mem u, Scal v, cl_mem res) {
  Launch("scalar_add", nw, u, v, res);
}

void OpenCL::Sub(MSize nw, cl_mem u, Scal v, cl_mem res) {
  Launch("scalar_sub", nw, u, v, res);
}
void OpenCL::Sub(MSize nw, Scal u, cl_mem v, cl_mem res) {
  Launch("scalar_sub2", nw, u, v, res);
}

void OpenCL::Mul(MSize nw, cl_mem u, Scal v, cl_mem res) {
  Launch("scalar_mul", nw, u, v, res);
}

void OpenCL::Div(MSize nw, cl_mem u, Scal v, cl_mem res) {
  Launch("scalar_div", nw, u, v, res);
}

void OpenCL::Div(MSize nw, Scal u, cl_mem v, cl_mem res) {
  Launch("scalar_div2", nw, u, v, res);
}

void OpenCL::Sin(MSize nw, cl_mem u, cl_mem res) {
  Launch("unary_sin", nw, u, res);
}

void OpenCL::Cos(MSize nw, cl_mem u, cl_mem res) {
  Launch("unary_cos", nw, u, res);
}

void OpenCL::Exp(MSize nw, cl_mem u, cl_mem res) {
  Launch("unary_exp", nw, u, res);
}

void OpenCL::Log(MSize nw, cl_mem u, cl_mem res) {
  Launch("unary_log", nw, u, res);
}

void OpenCL::Sqr(MSize nw, cl_mem u, cl_mem res) {
  Launch("unary_sqr", nw, u, res);
}

void OpenCL::Sqrt(MSize nw, cl_mem u, cl_mem res) {
  Launch("unary_sqrt", nw, u, res);
}

void OpenCL::Roll(MSize nw, cl_mem u, MInt shift, cl_mem res) {
  shift[0] = shift[0] % nw[0];
  shift[1] = shift[1] % nw[1];
  if (shift[0] < 0) {
    shift[0] += nw[0];
  }
  if (shift[1] < 0) {
    shift[1] += nw[1];
  }
  Launch("unary_roll", nw, u, shift[0], shift[1], res);
}

void OpenCL::Restrict(MSize nw, cl_mem u, cl_mem res) {
  Launch("field_restrict", nw, u, res);
}

void OpenCL::RestrictAdjoint(MSize nw, cl_mem u, cl_mem res) {
  Launch("field_restrict_adjoint", nw, u, res);
}

void OpenCL::Interpolate(MSize nwf, cl_mem u, cl_mem res) {
  Launch("field_interpolate", nwf, u, res);
}

void OpenCL::InterpolateAdjoint(MSize nw, cl_mem u, cl_mem res) {
  Launch("field_interpolate_adjoint", nw, u, res);
}

void OpenCL::Conv(MSize nw, cl_mem u, Scal a, Scal axm, Scal axp, Scal aym,
                  Scal ayp, cl_mem res) {
  Launch("unary_conv", nw, u, a, axm, axp, aym, ayp, res);
}

void OpenCL::Add(MSize nw, cl_mem u, cl_mem v, cl_mem res) {
  Launch("field_add", nw, u, v, res);
}

void OpenCL::Sub(MSize nw, cl_mem u, cl_mem v, cl_mem res) {
  Launch("field_sub", nw, u, v, res);
}

void OpenCL::Mul(MSize nw, cl_mem u, cl_mem v, cl_mem res) {
  Launch("field_mul", nw, u, v, res);
}

void OpenCL::Div(MSize nw, cl_mem u, cl_mem v, cl_mem res) {
  Launch("field_div", nw, u, v, res);
}

// Instantiations.
template double OpenCL::ReadAt<double>(OpenCL::MSize, cl_mem, OpenCL::MInt);
template void OpenCL::WriteAt<double>(OpenCL::MSize, cl_mem, OpenCL::MInt,
                                      double);
