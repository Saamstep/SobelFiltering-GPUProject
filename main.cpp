#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
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

    // Convert to grayscale
    cv::Mat gray;
    if (img.channels() > 1)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img;

    SobelProcessor processor;

    // --- 3x3 Sobel ---
    cv::Mat result3x3(gray.size(), gray.type());
    auto start3x3 = std::chrono::high_resolution_clock::now();
    processor.sobel_cpu<3>(gray.data, result3x3.data, gray.cols, gray.rows);
    auto end3x3 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU 3x3 Sobel Time: "
              << std::chrono::duration<double, std::milli>(end3x3 - start3x3).count()
              << " ms\n";

    // --- 5x5 Sobel ---
    cv::Mat result5x5(gray.size(), gray.type());
    auto start5x5 = std::chrono::high_resolution_clock::now();
    processor.sobel_cpu<5>(gray.data, result5x5.data, gray.cols, gray.rows);
    auto end5x5 = std::chrono::high_resolution_clock::now();
    std::cout << "CPU 5x5 Sobel Time: "
              << std::chrono::duration<double, std::milli>(end5x5 - start5x5).count()
              << " ms\n";

    // Show results
    cv::imshow("Input", img);
    cv::imshow("Sobel 3x3", result3x3);
    cv::imshow("Sobel 5x5", result5x5);

    cv::imwrite("sobel_3x3.png", result3x3);
    cv::imwrite("sobel_5x5.png", result5x5);

    cv::waitKey(0);
    return 0;
}