#pragma once

#include <algorithm>
#include <array>
#include <iosfwd>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CL_TARGET_OPENCL_VERSION 200
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include "macros.h"

#define CLCALL(x)                                                     \
  do {                                                                \
    cl_int CLCALL_error;                                              \
    CLCALL_error = x;                                                 \
    if (CLCALL_error != CL_SUCCESS) {                                 \
      throw std::runtime_error(                                       \
          FILELINE + ": CL failed: " + std::to_string(CLCALL_error)); \
    }                                                                 \
  } while (0)

struct OpenCL {
  static std::string GetErrorMessage(cl_int error);

  using Scal = double;
  static constexpr int kDim = 2;
  using MSize = std::array<size_t, kDim>;

  struct Config {
    int platform = 0;
    int verbose = 0;
    MSize global_size;  // Global array size.
  };

  struct Device {
    Device() = default;
    Device(const Device&) = delete;
    Device(Device&&) = default;
    Device& operator=(Device&) = delete;
    Device& operator=(Device&&) = default;
    struct PlatformInfo {
      cl_platform_id id;
      std::string name;
      std::string vendor;
    };
    static std::vector<PlatformInfo> GetPlatformInfos();
    struct DeviceInfo {
      cl_device_id id;
      std::string name;
      std::string extensions;
      size_t max_work_size;
    };
    static cl_device_id GetDevice(cl_platform_id platform);
    static DeviceInfo GetDeviceInfo(cl_platform_id platform);
    void Create(size_t pid);
    ~Device();
    operator cl_device_id() const {
      return handle;
    }
    cl_device_id handle = NULL;
    cl_platform_id platform = NULL;
  };

  struct Context {
    Context() = default;
    Context(const Context&) = delete;
    Context(Context&&) = default;
    Context& operator=(Context&) = delete;
    Context& operator=(Context&&) = default;
    void Create(const cl_device_id* device);
    ~Context();
    operator cl_context() const {
      return handle;
    }
    cl_context handle = NULL;
  };

  struct Queue {
    Queue() = default;
    Queue(const Queue&) = delete;
    Queue(Queue&&) = default;
    Queue& operator=(Queue&) = delete;
    Queue& operator=(Queue&&) = default;
    void Create(cl_context context, cl_device_id device);
    void Finish();
    ~Queue();
    operator cl_command_queue() const {
      return handle;
    }
    cl_command_queue handle = NULL;
  };

  struct Program {
    Program() = default;
    Program(const Program&) = delete;
    Program(Program&&) = default;
    Program& operator=(Program&) = delete;
    Program& operator=(Program&&) = default;
    void CreateFromString(std::string source, cl_context context,
                          cl_device_id device);
    void CreateFromStream(std::istream& in, cl_context context,
                          cl_device_id device);
    void CreateFromFile(std::string source_path, cl_context context,
                        cl_device_id device);
    operator cl_program() const {
      return handle;
    }
    ~Program();
    cl_program handle = NULL;
  };
  template <class T>
  struct Buffer {
    Buffer() = default;
    Buffer(cl_context context, size_t size,
           cl_mem_flags flags = CL_MEM_READ_WRITE) {
      Create(context, size, flags);
    }
    Buffer(const Buffer&) = delete;
    Buffer(Buffer&& other) : handle(other.handle), size_(other.size_) {
      other.handle = NULL;
    }
    Buffer& operator=(Buffer&) = delete;
    Buffer& operator=(Buffer&& other) = delete;
    void Create(cl_context context, size_t size,
                cl_mem_flags flags = CL_MEM_READ_WRITE) {
      size_ = size;
      cl_int error;
      handle = clCreateBuffer(context, flags, sizeof(T) * size_, NULL, &error);
      CLCALL(error);
    }
    ~Buffer() {
      if (handle) {
        clReleaseMemObject(handle);
      }
    }
    size_t size() const {
      return size_;
    }
    void EnqueueRead(cl_command_queue queue, T* buf) const {
      CLCALL(clEnqueueReadBuffer(queue, handle, CL_TRUE, 0, sizeof(T) * size_,
                                 buf, 0, NULL, NULL));
    }
    void EnqueueRead(cl_command_queue queue, std::vector<T>& buf) const {
      EnqueueRead(queue, buf.data());
    }
    void EnqueueWrite(cl_command_queue queue, const T* buf) {
      CLCALL(clEnqueueWriteBuffer(queue, handle, CL_TRUE, 0, sizeof(T) * size_,
                                  buf, 0, NULL, NULL));
    }
    void EnqueueWrite(cl_command_queue queue, const std::vector<T>& buf) {
      EnqueueWrite(queue, buf.data());
    }
    void EnqueueWriteBuffer(cl_command_queue queue, const Buffer& src) {
      fassert_equal(src.size_, size_);
      CLCALL(clEnqueueCopyBuffer(queue, src, handle, 0, 0, sizeof(T) * size_, 0,
                                 NULL, NULL));
    }
    operator cl_mem() const {
      return handle;
    }
    void swap(Buffer& other) {
      std::swap(handle, other.handle);
      std::swap(size_, other.size_);
    }
    cl_mem handle = NULL;
    size_t size_;
  };

  // Buffer that allocates memory both on device and on host.
  // Memory on host is accessed by operator[].
  template <class T>
  struct MirroredBuffer : public Buffer<T> {
    using Base = Buffer<T>;
    using Base::handle;
    void Create(cl_context context, size_t size,
                cl_mem_flags flags = CL_MEM_READ_WRITE) {
      Base::Create(context, size, flags);
      buf.resize(this->size_);
    }
    void EnqueueRead(cl_command_queue queue) {
      Base::EnqueueRead(queue, buf);
    }
    void EnqueueWrite(cl_command_queue queue) {
      Base::EnqueueWrite(queue, buf);
    }
    const T& operator[](size_t i) const {
      return buf[i];
    }
    T& operator[](size_t i) {
      return buf[i];
    }
    const T* data() const {
      return buf.data();
    }
    T* data() {
      return buf.data();
    }

    std::vector<T> buf;
  };

  struct Kernel {
    Kernel() = default;
    Kernel(const Kernel&) = delete;
    Kernel(Kernel&&) = default;
    Kernel& operator=(Kernel&) = delete;
    Kernel& operator=(Kernel&&) = default;
    void Create(cl_program program, std::string name_);
    ~Kernel();
    template <class T>
    void SetArg(int pos, const T& value) {
      cl_int error = clSetKernelArg(handle, pos, sizeof(T), &value);
      fassert_equal(error, CL_SUCCESS,
                    ". SetArg failed for kernel '" + name + "' at position " +
                        std::to_string(pos));
    }
    template <class T>
    void SetArg(int pos, const Buffer<T>& value) {
      SetArg(pos, value.handle);
    }
    template <class T>
    void SetArg(int pos, const MirroredBuffer<T>& value) {
      SetArg(pos, value.handle);
    }
    void Enqueue(cl_command_queue queue, MSize global, MSize local) {
      CLCALL(clEnqueueNDRangeKernel(queue, handle, kDim, NULL, global.data(),
                                    local.data(), 0, NULL, NULL));
    }
    void EnqueueWithArgsImpl(int, cl_command_queue queue, MSize global,
                             MSize local) {
      Enqueue(queue, global, local);
    }
    template <class T, class... Args>
    void EnqueueWithArgsImpl(int pos, cl_command_queue queue, MSize global,
                             MSize local, const T& value, const Args&... args) {
      SetArg(pos, value);
      EnqueueWithArgsImpl(pos + 1, queue, global, local, args...);
    }
    template <class... Args>
    void EnqueueWithArgs(cl_command_queue queue, MSize global, MSize local,
                         const Args&... args) {
      EnqueueWithArgsImpl(0, queue, global, local, args...);
    }

    operator cl_kernel() const {
      return handle;
    }
    cl_kernel handle = NULL;
    std::string name;
  };

  OpenCL(const Config&);
  OpenCL(OpenCL&&) = default;
  // Reduction operations.
  Scal Max(cl_mem u);
  Scal Min(cl_mem u);
  Scal Sum(cl_mem u);
  Scal Dot(cl_mem u, cl_mem v);
  // Element access.
  template <class T>
  T ReadAt(cl_mem u, int ix, int iy);
  template <class T>
  void WriteAt(cl_mem u, int ix, int iy, T value);
  // Assignment operations.
  void Fill(cl_mem u, Scal value);
  void AssignAdd(cl_mem u, cl_mem v);
  // Unary operations.
  void Add(cl_mem u, Scal v, cl_mem res);
  void Sub(cl_mem u, Scal v, cl_mem res);
  void Sub(Scal u, cl_mem v, cl_mem res);
  void Mul(cl_mem u, Scal v, cl_mem res);
  void Div(cl_mem u, Scal v, cl_mem res);
  void Div(Scal u, cl_mem v, cl_mem res);
  void Sin(cl_mem v, cl_mem res);
  void Cos(cl_mem v, cl_mem res);
  void Exp(cl_mem v, cl_mem res);
  void Log(cl_mem v, cl_mem res);
  // Binary operations.
  void Add(cl_mem u, cl_mem v, cl_mem res);
  void Sub(cl_mem u, cl_mem v, cl_mem res);
  void Mul(cl_mem u, cl_mem v, cl_mem res);
  void Div(cl_mem u, cl_mem v, cl_mem res);

  void LaunchImpl(int, Kernel& kernel) {
    kernel.Enqueue(queue_, global_size_, local_size_);
  }
  template <class T, class... Args>
  void LaunchImpl(int pos, Kernel& kernel, const T& value,
                  const Args&... args) {
    kernel.SetArg(pos, value);
    LaunchImpl(pos + 1, kernel, args...);
  }
  template <class... Args>
  void Launch(const std::string& name, const Args&... args) {
    auto it = kernels_.find(name);
    fassert(it != kernels_.end(), "Kernel '" + name + "' not found");
    LaunchImpl(0, it->second, args...);
  }

  // Accessors.
  Context& context() {
    return context_;
  }
  Queue& queue() {
    return queue_;
  }

  Context context_;
  Device device_;
  typename Device::DeviceInfo device_info_;
  Queue queue_;
  Program program_;
  std::map<std::string, Kernel> kernels_;
  MirroredBuffer<Scal> d_buf_reduce_;

  MSize global_size_;
  MSize local_size_;
  size_t ngroups_;
  int start_;   // offset of first element of inner cells
  int lead_y_;  // leading dimension in y, factor before y in linear index
};
