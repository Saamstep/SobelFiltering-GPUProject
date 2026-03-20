#include "sobel.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

/* ------------------------------
* HELPER FUNCTIONS
  --------------------------------- */
namespace
{
    /// @brief Had a lot of issues with parsing errors. This was suggested by AI in order to get error logs. Otherwise stdout would be completely blank!
    /// @param err
    /// @param msg
    void cuda_assert(cudaError_t err, const char *msg)
    {
        if (err != cudaSuccess)
        {
            std::cerr << msg << " failed: " << cudaGetErrorString(err) << '\n';
            throw std::runtime_error(msg);
        }
    }
}

/* ------------------------------
* KERNEL FUNCTIONS
  --------------------------------- */

// Simple 1D index for global kernel
__global__ void kernel_global(unsigned char *in, unsigned char *out, int w, int h, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int R = K / 2;

    if (x >= R && x < w - R && y >= R && y < h - R)
    {
        int gx = 0, gy = 0;
        for (int ky = -R; ky <= R; ky++)
        {
            for (int kx = -R; kx <= R; kx++)
            {
                int pixel = in[(y + ky) * w + (x + kx)];

                // Hardcode 3x3 or 5x5 kernels
                if (K == 3)
                {
                    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
                    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
                    gx += pixel * Gx[ky + R][kx + R];
                    gy += pixel * Gy[ky + R][kx + R];
                }
                else if (K == 5)
                {
                    const int Gx[5][5] = {{-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2}, {-4, -2, 0, 2, 4}, {-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2}};
                    const int Gy[5][5] = {{-2, -2, -4, -2, -2}, {-1, -1, -2, -1, -1}, {0, 0, 0, 0, 0}, {1, 1, 2, 1, 1}, {2, 2, 4, 2, 2}};
                    gx += pixel * Gx[ky + R][kx + R];
                    gy += pixel * Gy[ky + R][kx + R];
                }
            }
        }
        int mag = abs(gx) + abs(gy);
        out[y * w + x] = (mag > 255) ? 255 : mag;
    }
}

__global__ void kernel_shared(unsigned char *in, unsigned char *out, int w, int h, int K)
{
    // Shared memory version
    extern __shared__ unsigned char sdata[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int R = K / 2;

    if (x >= R && x < w - R && y >= R && y < h - R)
    {
        // Copy block to shared memory
        int block_w = blockDim.x + 2 * R;
        int block_h = blockDim.y + 2 * R;

        for (int by = ty; by < block_h; by += blockDim.y)
        {
            for (int bx = tx; bx < block_w; bx += blockDim.x)
            {
                int ix = blockIdx.x * blockDim.x + bx - R;
                int iy = blockIdx.y * blockDim.y + by - R;

                if (ix >= 0 && ix < w && iy >= 0 && iy < h)
                    sdata[by * block_w + bx] = in[iy * w + ix];
                else
                    sdata[by * block_w + bx] = 0;
            }
        }
        __syncthreads();

        // Compute sobel
        int gx = 0, gy = 0;
        for (int ky = -R; ky <= R; ky++)
            for (int kx = -R; kx <= R; kx++)
            {
                int pixel = sdata[(ty + ky + R) * block_w + (tx + kx + R)];

                if (K == 3)
                {
                    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
                    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
                    gx += pixel * Gx[ky + R][kx + R];
                    gy += pixel * Gy[ky + R][kx + R];
                }
                else if (K == 5)
                {
                    const int Gx[5][5] = {{-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2}, {-4, -2, 0, 2, 4}, {-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2}};
                    const int Gy[5][5] = {{-2, -2, -4, -2, -2}, {-1, -1, -2, -1, -1}, {0, 0, 0, 0, 0}, {1, 1, 2, 1, 1}, {2, 2, 4, 2, 2}};
                    gx += pixel * Gx[ky + R][kx + R];
                    gy += pixel * Gy[ky + R][kx + R];
                }
            }

        int mag = abs(gx) + abs(gy);
        out[y * w + x] = (mag > 255) ? 255 : mag;
    }
}

// Host wrappers
void SobelProcessor::cuda_setup()
{
    // Force CUDA runtime/context initialization to get more accurate benchmarked work starts.
    cuda_assert(cudaFree(nullptr), "cudaFree(0) warm-up");
    cuda_assert(cudaDeviceSynchronize(), "cuda warm-up sync");
}

void SobelProcessor::sobel_cuda_global(unsigned char *in, unsigned char *out, int w, int h, int K, int block_x, int block_y, CudaTimingBreakdown *timing)
{
    unsigned char *d_in = nullptr, *d_out = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr;
    size_t size = w * h * sizeof(unsigned char);
    cuda_assert(cudaMalloc(&d_in, size), "cudaMalloc(d_in)");
    cuda_assert(cudaMalloc(&d_out, size), "cudaMalloc(d_out)");
    cuda_assert(cudaStreamCreate(&stream), "cudaStreamCreate");
    cuda_assert(cudaEventCreate(&e0), "cudaEventCreate(e0)");
    cuda_assert(cudaEventCreate(&e1), "cudaEventCreate(e1)");
    cuda_assert(cudaEventCreate(&e2), "cudaEventCreate(e2)");
    cuda_assert(cudaEventCreate(&e3), "cudaEventCreate(e3)");

    cuda_assert(cudaEventRecord(e0, stream), "cudaEventRecord(e0)");
    cuda_assert(cudaMemcpyAsync(d_in, in, size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
    cuda_assert(cudaMemsetAsync(d_out, 0, size, stream), "cudaMemsetAsync(d_out)");

    dim3 block(block_x, block_y);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    cuda_assert(cudaEventRecord(e1, stream), "cudaEventRecord(e1)");
    kernel_global<<<grid, block, 0, stream>>>(d_in, d_out, w, h, K);
    cuda_assert(cudaGetLastError(), "sobel_kernel_global launch");
    cuda_assert(cudaEventRecord(e2, stream), "cudaEventRecord(e2)");
    cuda_assert(cudaMemcpyAsync(out, d_out, size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
    cuda_assert(cudaEventRecord(e3, stream), "cudaEventRecord(e3)");
    cuda_assert(cudaEventSynchronize(e3), "cudaEventSynchronize(e3)");

    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    cuda_assert(cudaEventElapsedTime(&h2d_ms, e0, e1), "cudaEventElapsedTime H2D");
    cuda_assert(cudaEventElapsedTime(&kernel_ms, e1, e2), "cudaEventElapsedTime kernel");
    cuda_assert(cudaEventElapsedTime(&d2h_ms, e2, e3), "cudaEventElapsedTime D2H");
    if (timing != nullptr)
    {
        timing->h2d_ms = h2d_ms;
        timing->kernel_ms = kernel_ms;
        timing->d2h_ms = d2h_ms;
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);
}

void SobelProcessor::sobel_cuda_shared(unsigned char *in, unsigned char *out, int w, int h, int K, int block_x, int block_y, CudaTimingBreakdown *timing)
{
    unsigned char *d_in = nullptr, *d_out = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr, e3 = nullptr;
    size_t size = w * h * sizeof(unsigned char);
    cuda_assert(cudaMalloc(&d_in, size), "cudaMalloc(d_in)");
    cuda_assert(cudaMalloc(&d_out, size), "cudaMalloc(d_out)");
    cuda_assert(cudaStreamCreate(&stream), "cudaStreamCreate");
    cuda_assert(cudaEventCreate(&e0), "cudaEventCreate(e0)");
    cuda_assert(cudaEventCreate(&e1), "cudaEventCreate(e1)");
    cuda_assert(cudaEventCreate(&e2), "cudaEventCreate(e2)");
    cuda_assert(cudaEventCreate(&e3), "cudaEventCreate(e3)");

    cuda_assert(cudaEventRecord(e0, stream), "cudaEventRecord(e0)");
    cuda_assert(cudaMemcpyAsync(d_in, in, size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
    cuda_assert(cudaMemsetAsync(d_out, 0, size, stream), "cudaMemsetAsync(d_out)");

    dim3 block(block_x, block_y);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    int radius = K / 2;
    size_t shared_mem = (block.x + 2 * radius) * (block.y + 2 * radius) * sizeof(unsigned char);
    cuda_assert(cudaEventRecord(e1, stream), "cudaEventRecord(e1)");
    kernel_shared<<<grid, block, shared_mem, stream>>>(d_in, d_out, w, h, K);
    cuda_assert(cudaGetLastError(), "sobel_kernel_shared launch");
    cuda_assert(cudaEventRecord(e2, stream), "cudaEventRecord(e2)");
    cuda_assert(cudaMemcpyAsync(out, d_out, size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
    cuda_assert(cudaEventRecord(e3, stream), "cudaEventRecord(e3)");
    cuda_assert(cudaEventSynchronize(e3), "cudaEventSynchronize(e3)");

    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    cuda_assert(cudaEventElapsedTime(&h2d_ms, e0, e1), "cudaEventElapsedTime H2D");
    cuda_assert(cudaEventElapsedTime(&kernel_ms, e1, e2), "cudaEventElapsedTime kernel");
    cuda_assert(cudaEventElapsedTime(&d2h_ms, e2, e3), "cudaEventElapsedTime D2H");
    if (timing != nullptr)
    {
        timing->h2d_ms = h2d_ms;
        timing->kernel_ms = kernel_ms;
        timing->d2h_ms = d2h_ms;
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);
}
