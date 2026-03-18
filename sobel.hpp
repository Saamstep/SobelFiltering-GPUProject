#pragma once
#include <opencv2/opencv.hpp>

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

#ifdef WIN32 // Lock to windows only systems - assume they have CUDA support
    void SobelProcessor::sobel_cuda_naive(unsigned char *in, unsigned char *out, int w, int h, int K);
    void SobelProcessor::sobel_cuda_shared(unsigned char *in, unsigned char *out, int w, int h, int K);
#endif
};