#include "opencl.h"

#include <array>
#include <fstream>
#include <iostream>
#include <limits>

static const char* kKernelSource =
#include "kernels.inc"
    ;

template <class Scal>
std::string OpenCL<Scal>::GetErrorMessage(cl_int error) {
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

template <class Scal>
auto OpenCL<Scal>::Device::GetPlatformInfos() -> std::vector<PlatformInfo> {
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

template <class Scal>
cl_device_id OpenCL<Scal>::Device::GetDevice(cl_platform_id platform) {
  cl_device_id device;
  cl_int error;
  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (error == CL_DEVICE_NOT_FOUND) {
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  }
  fassert(error == CL_SUCCESS, GetErrorMessage(error));
  return device;
}

template <class Scal>
auto OpenCL<Scal>::Device::GetDeviceInfo(cl_platform_id platform)
    -> DeviceInfo {
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

template <class Scal>
void OpenCL<Scal>::Device::Create(size_t pid) {
  auto infos = GetPlatformInfos();
  fassert(pid < infos.size(),  //
          "Invalid platform index " + std::to_string(pid) +
              ". Maximum valid index is " + std::to_string(infos.size() - 1));
  platform = infos[pid].id;
  handle = GetDevice(platform);
}

template <class Scal>
OpenCL<Scal>::Device::~Device() {
  if (handle) {
    clReleaseDevice(handle);
  }
}

template <class Scal>
void OpenCL<Scal>::Context::Create(const cl_device_id* device) {
  cl_int error;
  handle = clCreateContext(NULL, 1, device, NULL, NULL, &error);
  CLCALL(error);
}

template <class Scal>
OpenCL<Scal>::Context::~Context() {
  if (handle) {
    clReleaseContext(handle);
  }
}

template <class Scal>
void OpenCL<Scal>::Queue::Create(cl_context context, cl_device_id device) {
  cl_int error;
  handle = clCreateCommandQueue(context, device, 0, &error);
  CLCALL(error);
}

template <class Scal>
void OpenCL<Scal>::Queue::Finish() {
  CLCALL(clFinish(handle));
}

template <class Scal>
OpenCL<Scal>::Queue::~Queue() {
  if (handle) {
    clReleaseCommandQueue(handle);
  }
}

template <class Scal>
void OpenCL<Scal>::Program::CreateFromString(std::string source,
                                             cl_context context,
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

template <class Scal>
void OpenCL<Scal>::Program::CreateFromStream(std::istream& in,
                                             cl_context context,
                                             cl_device_id device) {
  std::stringstream ss;
  ss << in.rdbuf();
  CreateFromString(ss.str(), context, device);
}

template <class Scal>
void OpenCL<Scal>::Program::CreateFromFile(std::string source_path,
                                           cl_context context,
                                           cl_device_id device) {
  std::ifstream fin(source_path);
  CreateFromStream(fin, context, device);
}

template <class Scal>
OpenCL<Scal>::Program::~Program() {
  if (handle) {
    clReleaseProgram(handle);
  }
}

template <class Scal>
void OpenCL<Scal>::Kernel::Create(cl_program program, std::string name_) {
  name = name_;
  cl_int error;
  handle = clCreateKernel(program, name.c_str(), &error);
  fassert_equal(error, CL_SUCCESS, ". Kernel '" + name + "' not found");
}

template <class Scal>
OpenCL<Scal>::Kernel::~Kernel() {
  if (handle) {
    clReleaseKernel(handle);
  }
}

template <class Scal>
OpenCL<Scal>::OpenCL(const Config& config) : global_size_(config.global_size) {
  device_.Create(config.platform);
  device_info_ = Device::GetDeviceInfo(device_.platform);

  fassert_equal(kDim, 2);
  auto prod = [](MSize s) { return s[0] * s[1]; };
  local_size_[0] = global_size_[0];
  local_size_[1] = global_size_[1];
  while (prod(local_size_) > device_info_.max_work_size) {
    fassert(local_size_[0] % 2 == 0);
    fassert(local_size_[1] % 2 == 0);
    local_size_[0] /= 2;
    local_size_[1] /= 2;
  }
  ngroups_ = prod(global_size_) / prod(local_size_);
  start_ = 0;
  lead_y_ = global_size_[0];

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
  d_buf_reduce_.Create(context_, ngroups_, CL_MEM_WRITE_ONLY);
  kernel_max_.Create(program_, "field_max");
  kernel_dot_.Create(program_, "field_dot");
  kernel_sum_.Create(program_, "field_sum");
}

template <class Scal>
auto OpenCL<Scal>::Max(cl_mem d_u) -> Scal {
  kernel_max_.EnqueueWithArgs(queue_, global_size_, local_size_, start_,
                              lead_y_, d_u, d_buf_reduce_);
  d_buf_reduce_.EnqueueRead(queue_);
  queue_.Finish();
  Scal res = -std::numeric_limits<Scal>::max();
  for (size_t i = 0; i < ngroups_; ++i) {
    res = std::max(res, d_buf_reduce_[i]);
  }
  return res;
}

template <class Scal>
auto OpenCL<Scal>::Sum(cl_mem d_u) -> Scal {
  kernel_sum_.EnqueueWithArgs(queue_, global_size_, local_size_, start_,
                              lead_y_, d_u, d_buf_reduce_);
  d_buf_reduce_.EnqueueRead(queue_);
  queue_.Finish();
  Scal res = 0;
  for (size_t i = 0; i < ngroups_; ++i) {
    res += d_buf_reduce_[i];
  }
  return res;
}

template <class Scal>
auto OpenCL<Scal>::Dot(cl_mem d_u, cl_mem d_v) -> Scal {
  kernel_dot_.EnqueueWithArgs(queue_, global_size_, local_size_, start_,
                              lead_y_, d_u, d_v, d_buf_reduce_);
  d_buf_reduce_.EnqueueRead(queue_);
  queue_.Finish();
  Scal res = 0;
  for (size_t i = 0; i < ngroups_; ++i) {
    res += d_buf_reduce_[i];
  }
  return res;
}
