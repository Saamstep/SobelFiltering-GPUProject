# Sobel Filtering: CPU and CUDA

This project compares a CPU Sobel implementation against two CUDA versions:

- CPU Sobel with 3x3 and 5x5 kernels
- CUDA global Sobel using global memory
- CUDA shared-memory Sobel using a tiled shared buffer

The application loads an image with OpenCV, converts it to grayscale, runs the available Sobel paths, prints timing to the terminal, and displays the results in OpenCV windows.

## Current Status

Implemented now:

- `sobel_cpu` executable
- `sobel_gpu` executable
- CPU 3x3 timing with `std::chrono`
- CPU 5x5 timing with `std::chrono`
- CUDA global 3x3 timing with `std::chrono`
- CUDA shared 5x5 timing with `std::chrono`
- CUDA runtime error reporting in the host wrappers
- Post-build copy of required OpenCV DLLs on Windows

Notes:

- `sobel_cpu` is CPU-only and does not compile or link CUDA code.
- `sobel_gpu` compiles `main.cpp` and `sobel.cu`, and enables the CUDA code path with `SOBEL_ENABLE_CUDA`.
- The current GPU timings include host-to-device copy, kernel launch, synchronization, and device-to-host copy. They are end-to-end timings, not kernel-only timings.

## Project Files

- `main.cpp`: image loading, grayscale conversion, timing, and display
- `sobel.hpp`: CPU Sobel templates and the `SobelProcessor` interface
- `sobel.cu`: CUDA kernels and host wrapper functions
- `CMakeLists.txt`: build configuration for CPU and GPU targets

## Requirements

Windows is the actively configured platform in the current build.

- CMake 3.18+
- Visual Studio with MSVC
- OpenCV 4.x
- NVIDIA CUDA Toolkit
- NVIDIA GPU and compatible driver for the CUDA target

The current CMake setup expects OpenCV at:

```text
C:/opencv/build
```

Specifically, the Windows config uses:

```text
C:/opencv/build/x64/vc16/lib
```

## Build

Configure:

```powershell
& cmake -S . -B build -D SOBEL_CUDA_ARCHITECTURES=120-real
```

Build CPU target:

```powershell
& cmake --build build --config Release --target sobel_cpu
```

Build GPU target:

```powershell
& cmake --build build --config Release --target sobel_gpu
```

Notes:

- The project copies OpenCV runtime DLLs into the target output directory after build on Windows.
- `SOBEL_CUDA_ARCHITECTURES` should match the installed GPU. The current working configuration on this machine is `120-real` for an RTX 5080.
- If a build targets an older architecture such as `75`, the CUDA runtime may fall back to PTX JIT and fail with `the provided PTX was compiled with an unsupported toolchain` when the driver is older than the toolkit.

## Run

CPU:

```powershell
.\build\Release\sobel_cpu.exe <image-path>
```

GPU:

```powershell
.\build\Release\sobel_gpu.exe <image-path>
```

Expected behavior:

- The program prints timing results in the terminal.
- The program opens OpenCV display windows.
- The application waits on `cv::waitKey(0)`.

Current display behavior:

- `sobel_cpu` shows `CPU 3x3` and `CPU 5x5`
- `sobel_gpu` currently runs the CUDA paths and prints their timings, but `main.cpp` currently only displays the CPU result windows

## Timing

The current timings are measured in `main.cpp` with `std::chrono`.

Measured now:

- CPU 3x3
- CPU 5x5
- CUDA global 3x3
- CUDA shared 5x5

The CUDA timings are total call times from the host side. If kernel-only timing is needed, CUDA events should be added inside the CUDA path instead of relying on `std::chrono` around the wrapper calls.

## CUDA Implementation Notes

Current CUDA implementation details:

- CUDA global kernel launches one thread per pixel
- CUDA shared kernel loads a tile plus halo region into shared memory
- Both CUDA wrappers allocate device buffers, copy input to device, launch the kernel, synchronize, copy output back, and free memory
- CUDA wrapper functions now check launch and runtime errors and print failure messages to the terminal

## Limitations

- The project is currently configured around a Windows + Visual Studio + OpenCV + CUDA workflow
- OpenCV path detection is not generalized yet
- CUDA timings are not kernel-only
- There is no automated test suite in the repository
- The current application processes a single input image per run

## Output

Successful runs print lines similar to:

```text
CPU 3x3: 1.23 ms
CPU 5x5: 2.45 ms
CUDA Global 3x3: 0.80 ms
CUDA Shared 5x5: 0.62 ms
```

If a CUDA launch fails, the program now prints the CUDA error message to help diagnose architecture, driver, or runtime issues.
