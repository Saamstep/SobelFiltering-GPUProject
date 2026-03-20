#include <opencv2/opencv.hpp>

#include "sobel.hpp"

enum class Target
{
    CPU_3x3,
    CPU_5x5,
    CUDA_NAIVE,
    CUDA_SHARED
};

class SobelProcessor
{
public:
    // The main function to take an image, outputs an image
    cv::Mat process(const cv::Mat &input, Target target)
    {
        cv::Mat gray, output;

        // Ensure input is grayscale
        if (input.channels() > 1)
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else
            gray = input;

        output = cv::Mat::zeros(gray.size(), CV_8UC1);

        // Routing to different implementations
        switch (target)
        {
        case Target::CPU_3x3:
            run_cpu<3>(gray.data, output.data, gray.cols, gray.rows);
            break;
        case Target::CPU_5x5:
            run_cpu<5>(gray.data, output.data, gray.cols, gray.rows);
            break;
        case Target::CUDA_NAIVE:
            run_cuda_global(gray.data, output.data, gray.cols, gray.rows);
            break;
        case Target::CUDA_SHARED:
            // Example of window-sized processing
            run_cuda_shared(gray.data, output.data, gray.cols, gray.rows);
            break;
        }
        return output;
    }

private:
    void run_cpu(unsigned char *in, unsigned char *out, int w, int h);
    void run_cuda_global(unsigned char *in, unsigned char *out, int w, int h);
    void run_cuda_shared(unsigned char *in, unsigned char *out, int w, int h);
};