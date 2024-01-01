#include "opencl.h"

#include <fstream>
#include <iostream>

const char* kKernels =
#include "kernels.inc"
    ;

template <class M>
std::string OpenCL<M>::GetErrorMessage(cl_int error) {
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

template <class M>
auto OpenCL<M>::Device::GetPlatformInfos() -> std::vector<PlatformInfo> {
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

template <class M>
cl_device_id OpenCL<M>::Device::GetDevice(cl_platform_id platform) {
  cl_device_id device;
  cl_int error;
  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (error == CL_DEVICE_NOT_FOUND) {
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  }
  fassert(error == CL_SUCCESS, GetErrorMessage(error));
  return device;
}

template <class M>
auto OpenCL<M>::Device::GetDeviceInfo(cl_platform_id platform) -> DeviceInfo {
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

template <class M>
void OpenCL<M>::Device::Create(size_t pid) {
  auto infos = GetPlatformInfos();
  fassert(pid < infos.size(),
          util::Format("Invalid platform index {}. Maximum valid index is {}",
                       pid, infos.size() - 1));
  platform = infos[pid].id;
  handle = GetDevice(platform);
}

template <class M>
OpenCL<M>::Device::~Device() {
  if (handle) {
    clReleaseDevice(handle);
  }
}

template <class M>
void OpenCL<M>::Context::Create(const cl_device_id* device) {
  cl_int error;
  handle = clCreateContext(NULL, 1, device, NULL, NULL, &error);
  CLCALL(error);
}

template <class M>
OpenCL<M>::Context::~Context() {
  if (handle) {
    clReleaseContext(handle);
  }
}

template <class M>
void OpenCL<M>::Queue::Create(cl_context context, cl_device_id device) {
  cl_int error;
  handle = clCreateCommandQueue(context, device, 0, &error);
  CLCALL(error);
}

template <class M>
void OpenCL<M>::Queue::Finish() {
  CLCALL(clFinish(handle));
}

template <class M>
OpenCL<M>::Queue::~Queue() {
  if (handle) {
    clReleaseCommandQueue(handle);
  }
}

template <class M>
void OpenCL<M>::Program::CreateFromString(std::string source,
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

template <class M>
void OpenCL<M>::Program::CreateFromStream(std::istream& in, cl_context context,
                                          cl_device_id device) {
  std::stringstream ss;
  ss << in.rdbuf();
  CreateFromString(ss.str(), context, device);
}

template <class M>
void OpenCL<M>::Program::CreateFromFile(std::string source_path,
                                        cl_context context,
                                        cl_device_id device) {
  std::ifstream fin(source_path);
  CreateFromStream(fin, context, device);
}

template <class M>
OpenCL<M>::Program::~Program() {
  if (handle) {
    clReleaseProgram(handle);
  }
}

template <class M>
void OpenCL<M>::Kernel::Create(cl_program program, std::string name_) {
  name = name_;
  cl_int error;
  handle = clCreateKernel(program, name.c_str(), &error);
  fassert_equal(error, CL_SUCCESS, ". Kernel '" + name + "' not found");
}

template <class M>
OpenCL<M>::Kernel::~Kernel() {
  if (handle) {
    clReleaseKernel(handle);
  }
}

template <class M>
OpenCL<M>::OpenCL(const M& ms, const Vars& var) {
  auto ceil = [](MSize n, MSize d) {  //
    return (n + d - MSize(1)) / d * d;
  };

  const int pid = var.Int("opencl_platform", 0);
  device.Create(pid);
  auto dinfo = Device::GetDeviceInfo(device.platform);

  size = ms.GetIndexCells().size();
  msize = ms.GetInBlockCells().GetSize();
  local_size = MSize(1);
  while ((local_size * 2).prod() <= dinfo.max_work_size) {
    local_size *= 2;
  }
  global_size = ceil(MSize(msize), local_size);
  ngroups = global_size.prod() / local_size.prod();
  start = (*ms.Cells().begin()).raw();
  lead_y = ms.GetIndexCells().GetSize()[0];
  lead_z = ms.GetIndexCells().GetSize()[0] * ms.GetIndexCells().GetSize()[1];

  if (var.Int("opencl_verbose", 0) && ms.IsRoot()) {
    auto pinfo = Device::GetPlatformInfos()[pid];
    std::cout << util::Format(
        "Platform name: {}\n"
        "Platform vendor: {}\n"
        "Device name: {}\n"
        "Device extensions:\n{}\n",
        pinfo.name, pinfo.vendor, dinfo.name, dinfo.extensions);
  }

  context.Create(&device.handle);
  program.CreateFromString(kKernels, context, device);
  queue.Create(context, device);

  d_buf_reduce.Create(context, ngroups, CL_MEM_WRITE_ONLY);

  kernel_max.Create(program, "field_max");
  kernel_dot.Create(program, "field_dot");
  kernel_sum.Create(program, "field_sum");
}

template <class M>
auto OpenCL<M>::Max(cl_mem d_u) -> Scal {
  kernel_max.EnqueueWithArgs(queue, global_size, local_size, start, lead_y,
                             lead_z, d_u, d_buf_reduce);
  d_buf_reduce.EnqueueRead(queue);
  queue.Finish();
  Scal res = -std::numeric_limits<Scal>::max();
  for (size_t i = 0; i < ngroups; ++i) {
    res = std::max(res, d_buf_reduce[i]);
  }
  return res;
}

template <class M>
auto OpenCL<M>::Sum(cl_mem d_u) -> Scal {
  kernel_sum.EnqueueWithArgs(queue, global_size, local_size, start, lead_y,
                             lead_z, d_u, d_buf_reduce);
  d_buf_reduce.EnqueueRead(queue);
  queue.Finish();
  Scal res = 0;
  for (size_t i = 0; i < ngroups; ++i) {
    res += d_buf_reduce[i];
  }
  return res;
}

template <class M>
auto OpenCL<M>::Dot(cl_mem d_u, cl_mem d_v) -> Scal {
  kernel_dot.EnqueueWithArgs(queue, global_size, local_size, start, lead_y,
                             lead_z, d_u, d_v, d_buf_reduce);
  d_buf_reduce.EnqueueRead(queue);
  queue.Finish();
  Scal res = 0;
  for (size_t i = 0; i < ngroups; ++i) {
    res += d_buf_reduce[i];
  }
  return res;
}
