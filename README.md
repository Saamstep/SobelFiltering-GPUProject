# Sobel Filtering with CUDA

This benchmarks Sobel edge detection on CPU and CUDA, with the GPU path as the main focus.

## Build

### Requirements

- Windows
- CMake 3.18+
- Visual Studio with MSVC
- OpenCV 4.x
- NVIDIA CUDA Toolkit
- NVIDIA GPU and compatible driver

The current CMake setup expects OpenCV at:

```text
C:/opencv/build
```

Configure:

```powershell
& cmake -S . -B build -D SOBEL_CUDA_ARCHITECTURES=120-real
```

Build:

```powershell
& cmake --build build --config Release --target sobel_gpu
```

Notes:

- The project currently builds `sobel_gpu` only.
  - Don't build `sobel_cpu` it will break!
- OpenCV runtime DLLs are copied into the output directory after build.
- `SOBEL_CUDA_ARCHITECTURES` should match the installed GPU.

## Run

### Image Benchmark

```powershell
.\build\Release\sobel_gpu.exe <image-path>
```

### Video Benchmark

```powershell
.\build\Release\sobel_gpu.exe <video-path>
```