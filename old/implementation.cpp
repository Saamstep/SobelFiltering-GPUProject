#include "sobel.hpp"

// void SobelProcessor::run_cpu(unsigned char *in, unsigned char *out, int w, int h)
// {
//     for (int y = 1; y < h - 1; y++)
//     {
//         for (int x = 1; x < w - 1; x++)
//         {
//             int idx = y * w + x;

//             int gx =
//                 -in[(y - 1) * w + (x - 1)] + in[(y - 1) * w + (x + 1)] +
//                 -2 * in[y * w + (x - 1)] + 2 * in[y * w + (x + 1)] +
//                 -in[(y + 1) * w + (x - 1)] + in[(y + 1) * w + (x + 1)];

//             int gy =
//                 -in[(y - 1) * w + (x - 1)] - 2 * in[(y - 1) * w + x] - in[(y - 1) * w + (x + 1)] +
//                 in[(y + 1) * w + (x - 1)] + 2 * in[(y + 1) * w + x] + in[(y + 1) * w + (x + 1)];

//             int mag = abs(gx) + abs(gy);

//             if (mag > 255)
//                 mag = 255;

//             out[idx] = (unsigned char)mag;
//         }
//     }
// }

template <int K>
void SobelProcessor::run_cpu(unsigned char *in, unsigned char *out, int w, int h)
{
    constexpr int R = K / 2;

    for (int y = R; y < h - R; y++)
    {
        for (int x = R; x < w - R; x++)
        {
            int gx = 0;
            int gy = 0;

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
            out[y * w + x] = (mag > 255) ? 255 : mag;
        }
    }
}

void SobelProcessor::run_cuda_naive(unsigned char *in, unsigned char *out, int w, int h)
{
    // Placeholder for CUDA naive implementation
}

void SobelProcessor::run_cuda_shared(unsigned char *in, unsigned char *out, int w, int h)
{
    // Placeholder for CUDA shared memory implementation
}