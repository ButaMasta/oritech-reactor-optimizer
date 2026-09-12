cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Enable both C++ and CUDA languages
project(ReactorOptimizer LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "/home/buta/Dev/libtorch")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find the Torch package using the provided path
set(Torch_DIR "/home/buta/Dev/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Define your source files (update with your actual file names)
add_executable(reactor_optimizer 
    src/main.cpp 
    src/core/cuda/reactor_kernel.cu
    src/rl/actor_critic.cpp
    src/rl/reactor_gym.cpp
    src/rl/rollout_buffer.cpp
)

target_include_directories(reactor_optimizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link the LibTorch libraries to your executable
target_link_libraries(reactor_optimizer "${TORCH_LIBRARIES}")

# g++ compiler flags.
target_compile_options(reactor_optimizer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -pthread>
)

# NVCC compiler flags.
target_compile_options(reactor_optimizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_property(TARGET reactor_optimizer PROPERTY CXX_STANDARD 20)