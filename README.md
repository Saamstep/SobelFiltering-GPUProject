# Custom Sobel Filtering: CPU vs. CUDA Performance Analysis

A high-performance C++/CUDA implementation of the Sobel Edge Detection operator. This project provides a common interface to compare various hardware targets, ranging from sequential CPU execution to optimized "Windowed" GPU kernels.

## 🎯 Project Goals
* **Implement** the Sobel gradient algorithm ($G = \sqrt{G_x^2 + G_y^2}$) from scratch.
* **Compare** performance across three distinct targets:
    1. **CPU**: Sequential baseline (Single-threaded).
    2. **CUDA Naive**: Parallel batch processing (Global Memory).
    3. **CUDA Tiled**: Windowed optimization (Shared Memory/Multiprocessing).
* **Analyze** memory bandwidth bottlenecks and PCIe transfer overhead.

---

## 🛠 Requirements

### System Dependencies
* **OS**: Linux (Ubuntu 20.04+) or Windows 10/11.
* **GPU**: NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer).
* **Drivers**: NVIDIA Driver supporting CUDA 11.0+.

### Toolchain
| Requirement | Recommended Version | Purpose |
| :--- | :--- | :--- |
| **CMake** | 3.18+ | Build system & cross-platform config |
| **GCC/G++** | 9.0+ | Host (CPU) compiler |
| **CUDA Toolkit** | 11.0+ | `nvcc` compiler & GPU libraries |
| **OpenCV** | 4.x | Image I/O and visualization |

---

## 📂 Implementation Details

### 1. CPU Baseline
The CPU implementation focuses on **Cache Locality**. By traversing pixels in row-major order, we minimize L1 cache misses.


### 2. CUDA Naive (Batch Processing)
Maps one thread to every pixel. This version is simple but limited by **Global Memory Bandwidth**, as each thread fetches 9 neighboring pixels from high-latency VRAM.

### 3. CUDA Tiled (Windowed Optimization)
This version treats the GPU cores as a multiprocessing grid. It loads a "Window" (e.g., $16 \times 16$ or $32 \times 32$) of pixels into **Shared Memory** (on-chip L1 cache).
* **Rangeable Window Sizes**: Supports testing different block dimensions to find the "sweet spot" for occupancy.
* **Reduced Redundancy**: Neighboring threads share pixel data, reducing global VRAM requests by up to 90%.


---

## 🚀 Building the Project

1. **Configure**:
   ```bash
   mkdir build && cd build
   cmake ..