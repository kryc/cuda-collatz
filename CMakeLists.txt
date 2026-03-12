cmake_minimum_required(VERSION 3.18)
project(collatz LANGUAGES CXX CUDA)

# ── Default to Release for optimised builds ──
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Standards ──
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# nvcc 12.4 supports up to sm_90. Use PTX (compute_90) for forward
# compatibility with newer architectures (e.g. sm_120 Blackwell).
# The driver will JIT-compile the PTX at load time.
# Override with: cmake -B build -DCUDA_ARCH=80
set(CUDA_ARCH "90" CACHE STRING "CUDA architecture (e.g. 80, 86, 89, 90)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# Allow nvcc to use relaxed constexpr (needed for some GTest headers)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# ── Google Test via FetchContent ──
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ── Big integer limb count (default 2 = 128-bit) ──
set(COLLATZ_N_LIMBS "2" CACHE STRING "Number of 64-bit limbs for BigUint (2=128-bit, 4=256-bit, etc.)")

# ── Production library ──
add_library(collatz_lib
  src/collatz_kernel.cu
)
target_include_directories(collatz_lib PUBLIC include/)
target_compile_definitions(collatz_lib PUBLIC COLLATZ_N_LIMBS=${COLLATZ_N_LIMBS})
set_target_properties(collatz_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ── Main executable ──
add_executable(collatz
  src/main.cu
)
target_link_libraries(collatz PRIVATE collatz_lib)
set_target_properties(collatz PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# ── Test executable ──
add_executable(collatz_tests
  tests/test_big_uint.cu
  tests/test_collatz_kernel.cu
)
target_link_libraries(collatz_tests
  PRIVATE collatz_lib GTest::gtest_main
)
set_target_properties(collatz_tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
include(GoogleTest)
gtest_discover_tests(collatz_tests DISCOVERY_TIMEOUT 30)
