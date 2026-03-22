#include "sobel.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace
{
    void cuda_assert(cudaError_t err, const char *msg)
    {
        if (err != cudaSuccess)
        {
            std::cerr << msg << " failed: " << cudaGetErrorString(err) << '\n';
            throw std::runtime_error(msg);
        }
    }

    cudaStream_t as_stream(void *stream)
    {
        return reinterpret_cast<cudaStream_t>(stream);
    }

    cudaEvent_t as_event(void *event)
    {
        return reinterpret_cast<cudaEvent_t>(event);
    }
}

/* ------------------------------
* KERNEL FUNCTIONS
  --------------------------------- */

__global__ void kernel_global(unsigned char *in, unsigned char *out, int w, int h, int K)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int R = K / 2;

    if (x >= R && x < w - R && y >= R && y < h - R)
    {
        int gx = 0;
        int gy = 0;
        for (int ky = -R; ky <= R; ky++)
        {
            for (int kx = -R; kx <= R; kx++)
            {
                const int pixel = in[(y + ky) * w + (x + kx)];

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
        const int mag = abs(gx) + abs(gy);
        out[y * w + x] = (mag > 255) ? 255 : mag;
    }
}

__global__ void kernel_shared(unsigned char *in, unsigned char *out, int w, int h, int K)
{
    extern __shared__ unsigned char sdata[];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int R = K / 2;

    if (x >= R && x < w - R && y >= R && y < h - R)
    {
        const int block_w = blockDim.x + 2 * R;
        const int block_h = blockDim.y + 2 * R;

        for (int by = ty; by < block_h; by += blockDim.y)
        {
            for (int bx = tx; bx < block_w; bx += blockDim.x)
            {
                const int ix = blockIdx.x * blockDim.x + bx - R;
                const int iy = blockIdx.y * blockDim.y + by - R;

                if (ix >= 0 && ix < w && iy >= 0 && iy < h)
                    sdata[by * block_w + bx] = in[iy * w + ix];
                else
                    sdata[by * block_w + bx] = 0;
            }
        }
        __syncthreads();

        int gx = 0;
        int gy = 0;
        for (int ky = -R; ky <= R; ky++)
        {
            for (int kx = -R; kx <= R; kx++)
            {
                const int pixel = sdata[(ty + ky + R) * block_w + (tx + kx + R)];

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

        const int mag = abs(gx) + abs(gy);
        out[y * w + x] = (mag > 255) ? 255 : mag;
    }
}

__global__ void kernel_bgr_to_gray(const unsigned char *in, unsigned char *out, int w, int h, int channels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
        return;

    const int gray_idx = y * w + x;
    if (channels == 1)
    {
        out[gray_idx] = in[gray_idx];
        return;
    }

    const int color_idx = gray_idx * channels;
    const unsigned char b = in[color_idx + 0];
    const unsigned char g = in[color_idx + 1];
    const unsigned char r = in[color_idx + 2];
    out[gray_idx] = static_cast<unsigned char>((29 * b + 150 * g + 77 * r + 128) >> 8);
}

namespace
{
    void launch_sobel_kernel(bool use_shared,
                             unsigned char *d_in,
                             unsigned char *d_out,
                             int w,
                             int h,
                             int K,
                             int block_x,
                             int block_y,
                             cudaStream_t stream)
    {
        const dim3 block(block_x, block_y);
        const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

        if (use_shared)
        {
            const int radius = K / 2;
            const size_t shared_mem = (block.x + 2 * radius) * (block.y + 2 * radius) * sizeof(unsigned char);
            kernel_shared<<<grid, block, shared_mem, stream>>>(d_in, d_out, w, h, K);
            cuda_assert(cudaGetLastError(), "sobel_kernel_shared launch");
            return;
        }

        kernel_global<<<grid, block, 0, stream>>>(d_in, d_out, w, h, K);
        cuda_assert(cudaGetLastError(), "sobel_kernel_global launch");
    }

    void run_sobel_serial(unsigned char *host_in,
                          unsigned char *host_out,
                          int w,
                          int h,
                          int channels,
                          int K,
                          int block_x,
                          int block_y,
                          bool use_shared,
                          bool use_gpu_grayscale,
                          CudaTiming_t *timing)
    {
        unsigned char *d_color = nullptr;
        unsigned char *d_gray = nullptr;
        unsigned char *d_out = nullptr;
        cudaStream_t stream = nullptr;
        cudaEvent_t e0 = nullptr;
        cudaEvent_t e1 = nullptr;
        cudaEvent_t e2 = nullptr;
        cudaEvent_t e3 = nullptr;
        cudaEvent_t e4 = nullptr;
        const size_t gray_size = static_cast<size_t>(w) * h * sizeof(unsigned char);
        const size_t color_size = static_cast<size_t>(w) * h * channels * sizeof(unsigned char);

        if (use_gpu_grayscale)
            cuda_assert(cudaMalloc(&d_color, color_size), "cudaMalloc(d_color)");
        cuda_assert(cudaMalloc(&d_gray, gray_size), "cudaMalloc(d_gray)");
        cuda_assert(cudaMalloc(&d_out, gray_size), "cudaMalloc(d_out)");
        cuda_assert(cudaStreamCreate(&stream), "cudaStreamCreate");
        cuda_assert(cudaEventCreate(&e0), "cudaEventCreate(e0)");
        cuda_assert(cudaEventCreate(&e1), "cudaEventCreate(e1)");
        cuda_assert(cudaEventCreate(&e2), "cudaEventCreate(e2)");
        cuda_assert(cudaEventCreate(&e3), "cudaEventCreate(e3)");
        cuda_assert(cudaEventCreate(&e4), "cudaEventCreate(e4)");

        const dim3 block(block_x, block_y);
        const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

        cuda_assert(cudaEventRecord(e0, stream), "cudaEventRecord(e0)");
        if (use_gpu_grayscale)
            cuda_assert(cudaMemcpyAsync(d_color, host_in, color_size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D color");
        else
            cuda_assert(cudaMemcpyAsync(d_gray, host_in, gray_size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D gray");
        cuda_assert(cudaMemsetAsync(d_out, 0, gray_size, stream), "cudaMemsetAsync(d_out)");
        cuda_assert(cudaEventRecord(e1, stream), "cudaEventRecord(e1)");

        if (use_gpu_grayscale)
        {
            kernel_bgr_to_gray<<<grid, block, 0, stream>>>(d_color, d_gray, w, h, channels);
            cuda_assert(cudaGetLastError(), "bgr_to_gray launch");
        }
        cuda_assert(cudaEventRecord(e2, stream), "cudaEventRecord(e2)");

        launch_sobel_kernel(use_shared, d_gray, d_out, w, h, K, block_x, block_y, stream);
        cuda_assert(cudaEventRecord(e3, stream), "cudaEventRecord(e3)");
        cuda_assert(cudaMemcpyAsync(host_out, d_out, gray_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
        cuda_assert(cudaEventRecord(e4, stream), "cudaEventRecord(e4)");
        cuda_assert(cudaEventSynchronize(e4), "cudaEventSynchronize(e4)");

        float h2d_ms = 0.0f;
        float grayscale_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;
        cuda_assert(cudaEventElapsedTime(&h2d_ms, e0, e1), "cudaEventElapsedTime H2D");
        cuda_assert(cudaEventElapsedTime(&grayscale_ms, e1, e2), "cudaEventElapsedTime grayscale");
        cuda_assert(cudaEventElapsedTime(&kernel_ms, e2, e3), "cudaEventElapsedTime kernel");
        cuda_assert(cudaEventElapsedTime(&d2h_ms, e3, e4), "cudaEventElapsedTime D2H");
        if (timing != nullptr)
        {
            timing->h2d_ms = h2d_ms;
            timing->grayscale_ms = use_gpu_grayscale ? grayscale_ms : 0.0f;
            timing->kernel_ms = kernel_ms;
            timing->d2h_ms = d2h_ms;
        }

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaEventDestroy(e3);
        cudaEventDestroy(e4);
        cudaStreamDestroy(stream);
        cudaFree(d_color);
        cudaFree(d_gray);
        cudaFree(d_out);
    }
}

/* ------------------------------
* HOST WRAPPERS
  --------------------------------- */

void SobelProcessor::cuda_setup()
{
    cuda_assert(cudaFree(nullptr), "cudaFree(0) warm-up");
    cuda_assert(cudaDeviceSynchronize(), "cuda warm-up sync");
}

void SobelProcessor::sobel_cuda_global(unsigned char *in, unsigned char *out, int w, int h, int K, int block_x, int block_y, CudaTiming_t *timing)
{
    run_sobel_serial(in, out, w, h, 1, K, block_x, block_y, false, false, timing);
}

void SobelProcessor::sobel_cuda_shared(unsigned char *in, unsigned char *out, int w, int h, int K, int block_x, int block_y, CudaTiming_t *timing)
{
    run_sobel_serial(in, out, w, h, 1, K, block_x, block_y, true, false, timing);
}

void SobelProcessor::sobel_cuda_global_bgr(unsigned char *in, unsigned char *out, int w, int h, int channels, int K, int block_x, int block_y, CudaTiming_t *timing)
{
    run_sobel_serial(in, out, w, h, channels, K, block_x, block_y, false, true, timing);
}

void SobelProcessor::sobel_cuda_shared_bgr(unsigned char *in, unsigned char *out, int w, int h, int channels, int K, int block_x, int block_y, CudaTiming_t *timing)
{
    run_sobel_serial(in, out, w, h, channels, K, block_x, block_y, true, true, timing);
}

void SobelProcessor::cuda_pipeline_init(const CudaPipelineConfig_t &config, CudaPipelineContext_t &context)
{
    cuda_pipeline_destroy(context);

    context.config = config;
    context.slots.resize(config.ring_depth);

    const size_t gray_size = static_cast<size_t>(config.width) * config.height * sizeof(unsigned char);
    const size_t color_size = static_cast<size_t>(config.width) * config.height * config.channels * sizeof(unsigned char);

    for (int i = 0; i < config.ring_depth; ++i)
    {
        CudaPipelineSlot_t &slot = context.slots[i];
        cuda_assert(cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&slot.stream)), "cudaStreamCreate(slot)");
        cuda_assert(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&slot.event_start)), "cudaEventCreate(slot.event_start)");
        cuda_assert(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&slot.event_h2d_end)), "cudaEventCreate(slot.event_h2d_end)");
        cuda_assert(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&slot.event_gray_end)), "cudaEventCreate(slot.event_gray_end)");
        cuda_assert(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&slot.event_kernel_end)), "cudaEventCreate(slot.event_kernel_end)");
        cuda_assert(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&slot.event_d2h_end)), "cudaEventCreate(slot.event_d2h_end)");

        if (config.use_gpu_grayscale)
        {
            cuda_assert(cudaMalloc(&slot.d_color, color_size), "cudaMalloc(slot.d_color)");
            cuda_assert(cudaHostAlloc(reinterpret_cast<void **>(&slot.h_color), color_size, cudaHostAllocDefault), "cudaHostAlloc(slot.h_color)");
        }

        cuda_assert(cudaMalloc(&slot.d_gray, gray_size), "cudaMalloc(slot.d_gray)");
        cuda_assert(cudaMalloc(&slot.d_out, gray_size), "cudaMalloc(slot.d_out)");
        cuda_assert(cudaHostAlloc(reinterpret_cast<void **>(&slot.h_gray), gray_size, cudaHostAllocDefault), "cudaHostAlloc(slot.h_gray)");
        cuda_assert(cudaHostAlloc(reinterpret_cast<void **>(&slot.h_out), gray_size, cudaHostAllocDefault), "cudaHostAlloc(slot.h_out)");
    }
}

void SobelProcessor::cuda_pipeline_enqueue(CudaPipelineContext_t &context, int slot_index, const CudaPipelineFrameTiming_t &host_timing)
{
    CudaPipelineSlot_t &slot = context.slots[slot_index];
    cudaStream_t stream = as_stream(slot.stream);
    const CudaPipelineConfig_t &config = context.config;
    const size_t gray_size = static_cast<size_t>(config.width) * config.height * sizeof(unsigned char);
    const size_t color_size = static_cast<size_t>(config.width) * config.height * config.channels * sizeof(unsigned char);
    const dim3 block(config.block_x, config.block_y);
    const dim3 grid((config.width + block.x - 1) / block.x, (config.height + block.y - 1) / block.y);

    slot.pending_timing = host_timing;
    slot.frame_index = host_timing.frame_index;
    slot.in_flight = true;

    cuda_assert(cudaEventRecord(as_event(slot.event_start), stream), "cudaEventRecord(slot.event_start)");
    if (config.use_gpu_grayscale)
        cuda_assert(cudaMemcpyAsync(slot.d_color, slot.h_color, color_size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync pipeline H2D color");
    else
        cuda_assert(cudaMemcpyAsync(slot.d_gray, slot.h_gray, gray_size, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync pipeline H2D gray");

    cuda_assert(cudaMemsetAsync(slot.d_out, 0, gray_size, stream), "cudaMemsetAsync pipeline d_out");
    cuda_assert(cudaEventRecord(as_event(slot.event_h2d_end), stream), "cudaEventRecord(slot.event_h2d_end)");

    if (config.use_gpu_grayscale)
    {
        kernel_bgr_to_gray<<<grid, block, 0, stream>>>(slot.d_color, slot.d_gray, config.width, config.height, config.channels);
        cuda_assert(cudaGetLastError(), "pipeline bgr_to_gray launch");
    }
    cuda_assert(cudaEventRecord(as_event(slot.event_gray_end), stream), "cudaEventRecord(slot.event_gray_end)");

    launch_sobel_kernel(config.use_shared,
                        slot.d_gray,
                        slot.d_out,
                        config.width,
                        config.height,
                        config.kernel_size,
                        config.block_x,
                        config.block_y,
                        stream);
    cuda_assert(cudaEventRecord(as_event(slot.event_kernel_end), stream), "cudaEventRecord(slot.kernel_end)");
    cuda_assert(cudaMemcpyAsync(slot.h_out, slot.d_out, gray_size, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync pipeline D2H");
    cuda_assert(cudaEventRecord(as_event(slot.event_d2h_end), stream), "cudaEventRecord(slot.d2h_end)");
}

void SobelProcessor::cuda_pipeline_finalize(CudaPipelineContext_t &context, int slot_index, CudaPipelineFrameTiming_t &timing)
{
    CudaPipelineSlot_t &slot = context.slots[slot_index];
    if (!slot.in_flight)
    {
        timing = {};
        return;
    }

    cuda_assert(cudaEventSynchronize(as_event(slot.event_d2h_end)), "cudaEventSynchronize(slot.d2h_end)");

    float h2d_ms = 0.0f;
    float grayscale_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float gpu_total_ms = 0.0f;

    cuda_assert(cudaEventElapsedTime(&h2d_ms, as_event(slot.event_start), as_event(slot.event_h2d_end)), "cudaEventElapsedTime pipeline H2D");
    cuda_assert(cudaEventElapsedTime(&grayscale_ms, as_event(slot.event_h2d_end), as_event(slot.event_gray_end)), "cudaEventElapsedTime pipeline gray");
    cuda_assert(cudaEventElapsedTime(&kernel_ms, as_event(slot.event_gray_end), as_event(slot.event_kernel_end)), "cudaEventElapsedTime pipeline kernel");
    cuda_assert(cudaEventElapsedTime(&d2h_ms, as_event(slot.event_kernel_end), as_event(slot.event_d2h_end)), "cudaEventElapsedTime pipeline D2H");
    cuda_assert(cudaEventElapsedTime(&gpu_total_ms, as_event(slot.event_start), as_event(slot.event_d2h_end)), "cudaEventElapsedTime pipeline total");

    timing = slot.pending_timing;
    timing.h2d_ms = h2d_ms;
    timing.gpu_grayscale_ms = context.config.use_gpu_grayscale ? grayscale_ms : 0.0;
    timing.kernel_ms = kernel_ms;
    timing.d2h_ms = d2h_ms;
    timing.gpu_overhead_ms = std::max(0.0, static_cast<double>(gpu_total_ms) - timing.h2d_ms - timing.gpu_grayscale_ms - timing.kernel_ms - timing.d2h_ms);
    timing.total_latency_ms = timing.read_decode_ms +
                              timing.cpu_grayscale_ms +
                              timing.host_staging_ms +
                              timing.pipeline_wait_ms +
                              timing.h2d_ms +
                              timing.gpu_grayscale_ms +
                              timing.kernel_ms +
                              timing.d2h_ms +
                              timing.gpu_overhead_ms;

    slot.in_flight = false;
    slot.frame_index = -1;
    slot.pending_timing = {};
}

void SobelProcessor::cuda_pipeline_destroy(CudaPipelineContext_t &context)
{
    for (CudaPipelineSlot_t &slot : context.slots)
    {
        if (slot.in_flight && slot.event_d2h_end != nullptr)
            cudaEventSynchronize(as_event(slot.event_d2h_end));

        if (slot.event_start != nullptr)
            cudaEventDestroy(as_event(slot.event_start));
        if (slot.event_h2d_end != nullptr)
            cudaEventDestroy(as_event(slot.event_h2d_end));
        if (slot.event_gray_end != nullptr)
            cudaEventDestroy(as_event(slot.event_gray_end));
        if (slot.event_kernel_end != nullptr)
            cudaEventDestroy(as_event(slot.event_kernel_end));
        if (slot.event_d2h_end != nullptr)
            cudaEventDestroy(as_event(slot.event_d2h_end));
        if (slot.stream != nullptr)
            cudaStreamDestroy(as_stream(slot.stream));

        cudaFree(slot.d_color);
        cudaFree(slot.d_gray);
        cudaFree(slot.d_out);
        cudaFreeHost(slot.h_color);
        cudaFreeHost(slot.h_gray);
        cudaFreeHost(slot.h_out);

        slot = {};
    }

    context.slots.clear();
    context.config = {};
}
