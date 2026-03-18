#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "sobel.hpp"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: ./sobel_app image.jpg\n";
        return -1;
    }

    cv::Mat img = cv::imread(argv[1]);
    if (img.empty())
    {
        std::cout << "Could not load image\n";
        return -1;
    }

    cv::Mat gray;
    if (img.channels() > 1)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    SobelProcessor processor;

    // --- CPU 3x3 ---
    cv::Mat cpu3(gray.size(), gray.type());
    auto t1 = std::chrono::high_resolution_clock::now();
    processor.sobel_cpu<3>(gray.data, cpu3.data, gray.cols, gray.rows);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU 3x3: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

    // --- CPU 5x5 ---
    cv::Mat cpu5(gray.size(), gray.type());
    t1 = std::chrono::high_resolution_clock::now();
    processor.sobel_cpu<5>(gray.data, cpu5.data, gray.cols, gray.rows);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU 5x5: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

#ifdef WIN32
    // --- CUDA Naive 3x3 ---
    cv::Mat cuda3(gray.size(), gray.type());
    processor.sobel_cuda_naive(gray.data, cuda3.data, gray.cols, gray.rows, 3);

    // --- CUDA Shared 5x5 ---
    cv::Mat cuda5(gray.size(), gray.type());
    processor.sobel_cuda_shared(gray.data, cuda5.data, gray.cols, gray.rows, 5);

    cv::imshow("CUDA 3x3", cuda3);
    cv::imshow("CUDA 5x5", cuda5);
#endif

    cv::imshow("CPU 3x3", cpu3);
    cv::imshow("CPU 5x5", cpu5);
    cv::waitKey(0);

    return 0;
}