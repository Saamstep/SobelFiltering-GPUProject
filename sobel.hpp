#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Templated Sobel kernels
template <int K>
struct SobelKernel;

// 3x3 specialization
template <>
struct SobelKernel<3>
{
    static constexpr int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static constexpr int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
};

// 5x5 specialization
template <>
struct SobelKernel<5>
{
    static constexpr int Gx[5][5] = {
        {-2, -1, 0, 1, 2},
        {-2, -1, 0, 1, 2},
        {-4, -2, 0, 2, 4},
        {-2, -1, 0, 1, 2},
        {-2, -1, 0, 1, 2}};
    static constexpr int Gy[5][5] = {
        {-2, -2, -4, -2, -2},
        {-1, -1, -2, -1, -1},
        {0, 0, 0, 0, 0},
        {1, 1, 2, 1, 1},
        {2, 2, 4, 2, 2}};
};

struct CudaTiming_t
{
    float h2d_ms = 0.0f;
    float grayscale_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
};

struct CudaPipelineConfig_t
{
    int width = 0;
    int height = 0;
    int channels = 0;
    int kernel_size = 3;
    int block_x = 16;
    int block_y = 16;
    bool use_shared = false;
    bool use_gpu_grayscale = false;
    int ring_depth = 3;
};

struct CudaPipelineFrameTiming_t
{
    int frame_index = -1;
    double read_decode_ms = 0.0;
    double cpu_grayscale_ms = 0.0;
    double host_staging_ms = 0.0;
    double pipeline_wait_ms = 0.0;
    double h2d_ms = 0.0;
    double gpu_grayscale_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double gpu_overhead_ms = 0.0;
    double total_latency_ms = 0.0;
};

struct CudaPipelineSlot_t
{
    void *stream = nullptr;
    unsigned char *d_color = nullptr;
    unsigned char *d_gray = nullptr;
    unsigned char *d_out = nullptr;
    unsigned char *h_color = nullptr;
    unsigned char *h_gray = nullptr;
    unsigned char *h_out = nullptr;
    void *event_start = nullptr;
    void *event_h2d_end = nullptr;
    void *event_gray_end = nullptr;
    void *event_kernel_end = nullptr;
    void *event_d2h_end = nullptr;
    bool in_flight = false;
    int frame_index = -1;
    CudaPipelineFrameTiming_t pending_timing;
};

struct CudaPipelineContext_t
{
    CudaPipelineConfig_t config;
    std::vector<CudaPipelineSlot_t> slots;
};

class SobelProcessor
{
public:
    SobelProcessor() = default;

    // CPU template
    template <int K>
    void sobel_cpu(unsigned char *in, unsigned char *out, int w, int h)
    {
        constexpr int R = K / 2;
        for (int y = R; y < h - R; y++)
        {
            for (int x = R; x < w - R; x++)
            {
                int gx = 0, gy = 0;
                for (int ky = -R; ky <= R; ky++)
                {
                    for (int kx = -R; kx <= R; kx++)
                    {
                        int pixel = in[(y + ky) * w + (x + kx)];
                        gx += pixel * SobelKernel<K>::Gx[ky + R][kx + R];
                        gy += pixel * SobelKernel<K>::Gy[ky + R][kx + R];
                    }
                }
                int mag = abs(gx) + abs(gy);
                if (mag > 255)
                    mag = 255;
                out[y * w + x] = static_cast<unsigned char>(mag);
            }
        }
    }

    void cuda_setup();
    void sobel_cuda_global(unsigned char *in, unsigned char *out, int w, int h, int K, int block_x = 16, int block_y = 16, CudaTiming_t *timing = nullptr);
    void sobel_cuda_shared(unsigned char *in, unsigned char *out, int w, int h, int K, int block_x = 16, int block_y = 16, CudaTiming_t *timing = nullptr);
    void sobel_cuda_global_bgr(unsigned char *in, unsigned char *out, int w, int h, int channels, int K, int block_x = 16, int block_y = 16, CudaTiming_t *timing = nullptr);
    void sobel_cuda_shared_bgr(unsigned char *in, unsigned char *out, int w, int h, int channels, int K, int block_x = 16, int block_y = 16, CudaTiming_t *timing = nullptr);
    void cuda_pipeline_init(const CudaPipelineConfig_t &config, CudaPipelineContext_t &context);
    void cuda_pipeline_enqueue(CudaPipelineContext_t &context, int slot_index, const CudaPipelineFrameTiming_t &host_timing);
    void cuda_pipeline_finalize(CudaPipelineContext_t &context, int slot_index, CudaPipelineFrameTiming_t &timing);
    void cuda_pipeline_destroy(CudaPipelineContext_t &context);
};
