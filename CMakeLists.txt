cmake_minimum_required(VERSION 3.10.0)

project("autodiff" CXX)

option(BUILD_POISSON "Build poisson executable" ON)
option(BUILD_TESTS "Build tests" ON)
option(USE_WARNINGS "Enable compiler warnings" ON)
option(USE_OPENCL "Enable OpenCL" ON)

# Default build type.
set(BuildTypeValues None Debug Release RelWithDebInfo MinSizeRel)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: ${BuildTypeValues}." FORCE)
endif ()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${BuildTypeValues})

# C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (BUILD_POISSON)
  set(T "poisson")
  add_executable(poisson src/poisson.cpp)
  if (USE_OPENCL)
    add_executable(poisson_cl src/poisson_cl.cpp)
    target_link_libraries(poisson_cl opencl)
  endif()
endif()

if (BUILD_TESTS)
  add_executable(test_dual src/test_dual.cpp)
  add_executable(test_matrix src/test_matrix.cpp)
  add_executable(test_optimizer src/test_optimizer.cpp)
  add_executable(test_reverse src/test_reverse.cpp)
  if (USE_OPENCL)
    add_executable(test_opencl src/test_opencl.cpp)
    target_link_libraries(test_opencl opencl)
  endif()
endif()

if (USE_WARNINGS)
  add_compile_options(-Wall -Wextra -pedantic -Wshadow)
endif()

if (USE_OPENCL)
  set(T opencl)
  add_library(${T} src/opencl.cpp)

  set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_custom_command(
    OUTPUT  ${SRC}/kernels.inc
    COMMAND ./generate_inc kernels.cl kernels.inc
    WORKING_DIRECTORY ${SRC}
    DEPENDS ${SRC}/kernels.cl
  )

  add_custom_target(kernels_inc DEPENDS src/kernels.inc)
  set_source_files_properties(opencl.cpp PROPERTIES OBJECT_DEPENDS src/kernels.inc)
  add_dependencies(${T} kernels_inc)

  find_package(OpenCL)
  if (OpenCL_FOUND)
    target_link_libraries(${T} OpenCL::OpenCL)
  else()
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
      target_link_libraries(${T} CUDA::OpenCL)
    else()
      target_link_libraries(${T} OpenCL)
    endif()
  endif()
endif()