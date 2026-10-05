# SYCL Easy Tutorial

## Overview
This is an easy, example-driven SYCL tutorial written by jhson.
Feel free to read and use it. The examples favor readability over peak performance, so most of them are intentionally simple (and some are a little inefficient).

Each example focuses on one SYCL concept: device selection, buffers and accessors, `range` vs. `nd_range` kernels, local memory, and Unified Shared Memory (USM).
Every example runs its kernel on a GPU, measures the elapsed time, and checks the result against a CPU reference implementation.

## Project Structure
```
SYCL_easy_tutorial/
├── CMakeLists.txt              # Top-level build configuration (compiler path, SYCL flags)
├── include/                    # SYCL device kernels, written as C++ template functions
│   ├── deviceProperties.hpp
│   ├── vectorAddition.hpp
│   ├── matrixMultiplication.hpp
│   ├── matrixStencil.hpp
│   └── reduction.hpp
├── src/
│   └── deviceProperties.cpp    # Helpers that print device information
└── test/                       # Host programs that launch the kernels (one directory per example)
    ├── DeviceQuery/
    ├── VectorAddition/
    ├── MatrixMultiplication/
    ├── MatrixStencil/
    ├── Reduction/
    ├── Memory/
    └── ExecutionModel/         # Contains its own kernels in ExecutionModel/include/
```

- `include/` contains the device code, implemented as C++ template functions.
- `test/` contains the host code that prepares data, calls these kernels, and verifies the results.

## 0. How to Build and Run

### Tested environment
- clang 14.0.0 (Intel LLVM / DPC++, https://github.com/intel/llvm.git, commit `8c5b7017a925701ef4034056b5ed8e0fac2a0011`)
- CMake 3.20.2
- CUDA 10.2
- Intel CPU Runtime for OpenCL 18.1

### Configuration
The compiler location and SYCL options are hard-coded in the top-level `CMakeLists.txt`.
Before building, update them to match your own DPC++ installation:

```cmake
set(PATH_SYCL_BUILD /data/share/oneapi/llvm/build)   # Path to your intel/llvm build
set(CMAKE_CXX_COMPILER ${PATH_SYCL_BUILD}/bin/clang++)
set(SYCL_COMPILE_OPTION -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fopenmp)
```

By default the code is compiled for NVIDIA GPUs (`nvptx64-nvidia-cuda`).
To target a different backend, change `-fsycl-targets` (e.g. `spir64` for Intel/OpenCL devices).

### Build
```bash
mkdir build && cd build
cmake ..
make
```

### Run
Each example is built as a separate executable under `build/test/<Example>/`:

```bash
./test/DeviceQuery/DeviceQuery.out
./test/VectorAddition/VectorAddition.out
./test/MatrixMultiplication/MatrixMultiplication.out
./test/MatrixStencil/MatrixStencil.out
./test/Reduction/Reduction.out
./test/Memory/Memory.out
./test/ExecutionModel/ExecutionModel.out
```

> **Note:** Some examples use large inputs (several GB of memory), so they need a GPU with enough device memory.

## 1. Device Query Example
**Source:** `test/DeviceQuery/main.cpp`, `src/deviceProperties.cpp`

A very simple SYCL device query example.
It shows how to create a queue using the built-in device selectors (`host_selector`, `cpu_selector`, `gpu_selector`) and how to explicitly build a `platform` → `device` → `context` → `queue` chain.
For each device, it prints the name, vendor, and global memory size.

## 2. Vector Addition Example
**Source:** `include/vectorAddition.hpp`, `test/VectorAddition/main.cpp`

A very simple vector addition example: `C[i] = A[i] + B[i]` with 2^25 elements.
- Uses `sycl::buffer` and accessors for memory management.
- Each work item performs a single element-wise addition.
- No parallel optimization technique is applied.

## 3. Matrix Multiplication Example
**Source:** `include/matrixMultiplication.hpp`, `test/MatrixMultiplication/main.cpp`

A simple matrix multiplication example: `C[M,N] = A[M,K] * B[K,N]`.
- Launches a 1D `range` with one work item per output element.
- The work item for `C[m, n]` computes the dot product of the m-th row of A and the n-th column of B.
- No parallel optimization technique is applied.

## 4. Matrix Stencil Operation Example
**Source:** `include/matrixStencil.hpp`, `test/MatrixStencil/main.cpp`

A simple stencil operation on a matrix (a 2D correlation, as used in convolution layers).
- Applies a 3x3 filter to a 10240 x 10240 input with configurable stride (`offset`) and zero `pad`ding.
- Each work item computes one output element.
- No parallel optimization technique is applied.

## 5. Parallel Reduction Example
**Source:** `include/reduction.hpp`, `test/Reduction/main.cpp`

A parallel sum reduction over ~913M 64-bit integers (~6.8 GB).
- Uses an `nd_range` kernel and work-group **local memory** (`access::target::local`).
- Each work item accumulates `per_workitem` elements, which are strided by the total number of work items so that neighboring work items access contiguous memory.
- The kernel is launched repeatedly, shrinking the data each pass, until 256 or fewer partial sums remain; the host then adds up the remaining values.
- The result is compared against a sequential CPU sum, and both run times are printed.

The approach is based on the ideas in [1].

## 6. Memory Management Example
**Source:** `test/Memory/main.cpp`

SYCL provides two memory management models: **Unified Shared Memory (USM)** and **buffers**.
USM supports three kinds of allocations:

| Allocation | API | Location |
|---|---|---|
| Device | `sycl::malloc_device` | Device memory, explicit copies required |
| Host | `sycl::malloc_host` | Host memory, accessible from the device |
| Shared | `sycl::malloc_shared` | Migrates between host and device automatically |

This example repeats a copy-in → kernel → copy-out cycle (10 iterations, 6 GB per transfer) with each USM type and with buffers, and reports the average time of each, so you can compare their performance.

## 7. Execution Model Example
**Source:** `test/ExecutionModel/main.cpp`, `test/ExecutionModel/include/`

Compares SYCL kernel execution models using a large matrix multiplication (with USM device memory):
1. **Basic kernel** (`basicKernel.hpp`): a plain `parallel_for` over a 1D `range`, one work item per output element.
2. **ND-range kernel** (`NDRangeKernel.hpp`): a 2D `nd_range` with 16 x 16 work-groups.
3. **ND-range kernel, GPU-optimized** (`NDRangeKernel.hpp`): tiled matrix multiplication that loads tiles of A and B into **local memory** and synchronizes the work-group with `barrier`s, which reduces global memory traffic.
4. **Hierarchical kernel**: *TODO*

Define `DEBUG_MODE` in `main.cpp` to verify each result against a CPU implementation.

## References
[1] Harris, M. (2007). *Optimizing Parallel Reduction in CUDA*. NVIDIA Developer Technology.
