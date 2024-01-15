cmake_minimum_required(VERSION 3.10.0)

project("autodiff" CXX)

option(BUILD_POISSON "Build poisson executable" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_WASM "Build WebAssembly" OFF)
option(USE_WARNINGS "Enable compiler warnings" ON)
option(USE_OPENCL "Enable OpenCL" ON)
option(USE_MARCH_NATIVE "Compile with -march=native" ON)

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

include(CheckCXXCompilerFlag)
if (USE_MARCH_NATIVE)
  CHECK_CXX_COMPILER_FLAG("-march=native" CXX_MARCH_NATIVE)
  if(CXX_MARCH_NATIVE)
    add_compile_options(-march=native)
  else()
    message(SEND_ERROR "Unsupported -march=native. Setting USE_MARCH_NATIVE=OFF")
    set(USE_MARCH_NATIVE OFF CACHE BOOL "" FORCE)
  endif()
endif()

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${SRC})

set(T profiler)
add_library(${T} src/profiler.cpp)

if (BUILD_POISSON)
  set(T "poisson")
  add_executable(${T} src/poisson.cpp)
  if (USE_OPENCL)
    set(T "poisson_cl")
    add_executable(${T} src/poisson_cl.cpp)
    target_link_libraries(${T} opencl profiler)
  endif()
endif()

if (BUILD_TESTS)
  enable_testing()
  set(TESTDIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set(Tests test_dual test_matrix test_optimizer test_reverse)
  if (USE_OPENCL)
    list(APPEND Tests test_opencl)
  endif()
  foreach(T ${Tests})
    add_executable(${T} ${TESTDIR}/${T}.cpp)
    add_test(NAME ${T} COMMAND ${TESTDIR}/test ${CMAKE_BINARY_DIR}/${T} ${TESTDIR}/ref)
  endforeach()
  if (USE_OPENCL)
    target_link_libraries(test_opencl opencl profiler)
  endif()
endif()

if (USE_WARNINGS)
  add_compile_options(-Wall -Wextra -pedantic -Wshadow)
endif()

if (USE_OPENCL)
  set(T opencl)
  add_library(${T} src/opencl.cpp)

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

if (BUILD_WASM)
  if(BUILD_POISSON)
    message(SEND_ERROR "Option BUILD_WASM requires BUILD_POISSON=OFF")
    set(BUILD_POISSON OFF CACHE BOOL "" FORCE)
  endif()
  if(BUILD_TESTS)
    message(SEND_ERROR "Option BUILD_WASM requires BUILD_TESTS=OFF")
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
  endif()
  if(USE_OPENCL)
    message(SEND_ERROR "Option BUILD_WASM requires USE_OPENCL=OFF")
    set(USE_OPENCL OFF CACHE BOOL "" FORCE)
  endif()

  add_link_options(-sALLOW_MEMORY_GROWTH=1)

  # Suppress spurious warnings.
  add_compile_options(-Wno-dollar-in-identifier-extension)
  add_compile_options(-Wno-gnu-zero-variadic-macro-arguments)

  set(T graphics)
  add_library(${T} wasm/graphics.cpp)

  # Poisson.
  set(T poisson)
  add_executable(${T} wasm/poisson.cpp)
  target_link_libraries(${T} graphics)
  target_link_options(${T} PRIVATE "-sEXPORTED_FUNCTIONS=[\
    '_main', '_malloc' \
    , '_GetBitmapWidth', '_GetBitmapHeight' \
    , '_SendKeyDown', '_SendMouseMotion', '_SendMouseDown', '_SendMouseUp' \
    , '_Init', '_SetPause', '_GetStatusString' \
  ]")
  target_link_options(${T} PRIVATE "-sEXPORTED_RUNTIME_METHODS=['cwrap', 'ccall']")
  configure_file(wasm/poisson.html poisson.html COPYONLY)

  # Wave.
  set(T wave)
  add_executable(${T} wasm/wave.cpp)
  target_link_libraries(${T} graphics)
  target_link_options(${T} PRIVATE "-sEXPORTED_FUNCTIONS=[\
    '_main', '_malloc' \
    , '_GetBitmapWidth', '_GetBitmapHeight' \
    , '_SendKeyDown', '_SendMouseMotion', '_SendMouseDown', '_SendMouseUp' \
    , '_Init', '_SetPause', '_GetStatusString' \
  ]")
  target_link_options(${T} PRIVATE "-sEXPORTED_RUNTIME_METHODS=['cwrap', 'ccall']")
  configure_file(wasm/wave.html wave.html COPYONLY)

  # Heat.
  set(T heat)
  add_executable(${T} wasm/heat.cpp)
  target_link_libraries(${T} graphics)
  target_link_options(${T} PRIVATE "-sEXPORTED_FUNCTIONS=[\
    '_main', '_malloc' \
    , '_GetBitmapWidth', '_GetBitmapHeight' \
    , '_SendKeyDown', '_SendMouseMotion', '_SendMouseDown', '_SendMouseUp' \
    , '_Init', '_SetPause', '_GetStatusString' \
  ]")
  target_link_options(${T} PRIVATE "-sEXPORTED_RUNTIME_METHODS=['cwrap', 'ccall']")
  configure_file(wasm/heat.html heat.html COPYONLY)

  # Common.
  configure_file(wasm/index.html index.html COPYONLY)
  configure_file(wasm/favicon.png favicon.png COPYONLY)
  configure_file(wasm/view.css view.css COPYONLY)
  configure_file(wasm/view.js view.js COPYONLY)
  configure_file(wasm/assets/axes_xt.svg axes_xt.svg COPYONLY)
  configure_file(wasm/libs/buttons.js libs/buttons.js COPYONLY)
endif()

