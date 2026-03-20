#include <opencv2/opencv.hpp>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include "sobel.hpp"

namespace
{
    using Clock = std::chrono::high_resolution_clock;

    bool is_image_path(const std::string &path)
    {
        cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
        return !img.empty();
    }

    bool is_video_path(const std::string &path)
    {
        cv::VideoCapture cap(path);
        return cap.isOpened();
    }

    cv::Mat to_grayscale(const cv::Mat &frame)
    {
        cv::Mat gray;
        if (frame.channels() > 1)
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        else
            gray = frame;
        return gray;
    }

    void print_breakdown(const char *label, int k, int bx, int by, const CudaTimingBreakdown &timing)
    {
        std::cout << "  " << label << " breakdown (K=" << k << ", block=" << bx << "x" << by << "): "
                  << "H2D=" << timing.h2d_ms << " ms, "
                  << "Kernel=" << timing.kernel_ms << " ms, "
                  << "D2H=" << timing.d2h_ms << " ms\n";
    }

    struct VideoStats
    {
        double total_ms = 0.0;
        double h2d_ms = 0.0;
        double kernel_ms = 0.0;
        double d2h_ms = 0.0;
        int frames = 0;
    };

    struct VideoVariant
    {
        const char *label = "";
        int kernel_size = 3;
        bool use_shared = false;
        VideoStats stats;
    };

    template <typename Func>
    VideoStats benchmark_video_gpu(cv::VideoCapture &cap, SobelProcessor &processor, Func run_gpu)
    {
        VideoStats stats;
        cv::Mat frame;

        while (cap.read(frame))
        {
            cv::Mat gray = to_grayscale(frame);
            cv::Mat out(gray.size(), gray.type());
            CudaTimingBreakdown timing;

            auto t1 = Clock::now();
            run_gpu(gray, out, timing);
            auto t2 = Clock::now();

            stats.total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            stats.h2d_ms += timing.h2d_ms;
            stats.kernel_ms += timing.kernel_ms;
            stats.d2h_ms += timing.d2h_ms;
            stats.frames += 1;
        }

        return stats;
    }

    void print_video_stats(const char *label, const VideoStats &stats)
    {
        if (stats.frames == 0)
            return;

        double avg_total = stats.total_ms / stats.frames;
        double avg_h2d = stats.h2d_ms / stats.frames;
        double avg_kernel = stats.kernel_ms / stats.frames;
        double avg_d2h = stats.d2h_ms / stats.frames;
        double fps = 1000.0 / avg_total;

        std::cout << label << ": "
                  << stats.frames << " frames, "
                  << "avg total=" << avg_total << " ms, "
                  << "FPS=" << fps << '\n';
        std::cout << "  avg breakdown: "
                  << "H2D=" << avg_h2d << " ms, "
                  << "Kernel=" << avg_kernel << " ms, "
                  << "D2H=" << avg_d2h << " ms\n";
    }

    double average_total_ms(const VideoStats &stats)
    {
        if (stats.frames == 0)
            return std::numeric_limits<double>::infinity();
        return stats.total_ms / stats.frames;
    }

    template <typename Func>
    void preview_video_gpu(const std::string &path, const char *label, Func run_gpu)
    {
        cv::VideoCapture cap(path);
        if (!cap.isOpened())
            return;

        const double fps = cap.get(cv::CAP_PROP_FPS);
        const int delay_ms = fps > 0.0 ? std::max(1, static_cast<int>(1000.0 / fps)) : 33;

        cv::Mat frame;
        cv::Mat display;

        std::cout << "\nPreviewing " << label << " at source playback rate. Press Esc to close.\n";
        while (cap.read(frame))
        {
            cv::Mat gray = to_grayscale(frame);
            cv::Mat out(gray.size(), gray.type());
            CudaTimingBreakdown timing;
            run_gpu(gray, out, timing);

            cv::cvtColor(out, display, cv::COLOR_GRAY2BGR);
            cv::putText(display,
                        label,
                        cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cv::Scalar(0, 255, 0),
                        2);
            cv::imshow("GPU Video Preview", display);

            const int key = cv::waitKey(delay_ms);
            if (key == 27)
                break;
        }

        cv::destroyWindow("GPU Video Preview");
    }

    int run_image_benchmark(const std::string &path)
    {
        cv::Mat img = cv::imread(path);
        if (img.empty())
        {
            std::cout << "Could not load image\n";
            return -1;
        }

        cv::Mat gray = to_grayscale(img);
        SobelProcessor processor;
        const std::array<std::pair<int, int>, 5> block_sizes = {
            std::pair<int, int>{8, 8},
            std::pair<int, int>{16, 8},
            std::pair<int, int>{16, 16},
            std::pair<int, int>{32, 8},
            std::pair<int, int>{32, 16}};

        cv::Mat cpu3(gray.size(), gray.type());
        auto t1 = Clock::now();
        processor.sobel_cpu<3>(gray.data, cpu3.data, gray.cols, gray.rows);
        auto t2 = Clock::now();
        std::cout << "CPU 3x3: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        cv::Mat cpu5(gray.size(), gray.type());
        t1 = Clock::now();
        processor.sobel_cpu<5>(gray.data, cpu5.data, gray.cols, gray.rows);
        t2 = Clock::now();
        std::cout << "CPU 5x5: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        processor.cuda_setup();
        std::cout << "CUDA warm-up completed before timing.\n";

        cv::Mat cuda3(gray.size(), gray.type());
        CudaTimingBreakdown timing;

        t1 = Clock::now();
        processor.sobel_cuda_global(gray.data, cuda3.data, gray.cols, gray.rows, 3, 16, 16, &timing);
        t2 = Clock::now();
        print_breakdown("CUDA Global", 3, 16, 16, timing);
        std::cout << "CUDA Global 3x3: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared(gray.data, cuda3.data, gray.cols, gray.rows, 3, 16, 16, &timing);
        t2 = Clock::now();
        print_breakdown("CUDA Shared", 3, 16, 16, timing);
        std::cout << "CUDA Shared 3x3: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        cv::Mat cuda5(gray.size(), gray.type());

        t1 = Clock::now();
        processor.sobel_cuda_global(gray.data, cuda5.data, gray.cols, gray.rows, 5, 16, 16, &timing);
        t2 = Clock::now();
        print_breakdown("CUDA Global", 5, 16, 16, timing);
        std::cout << "CUDA Global 5x5: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared(gray.data, cuda5.data, gray.cols, gray.rows, 5, 16, 16, &timing);
        t2 = Clock::now();
        print_breakdown("CUDA Shared", 5, 16, 16, timing);
        std::cout << "CUDA Shared 5x5: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        std::cout << "\nBlock sweep (chrono total per call)\n";
        for (int i = 0; i < block_sizes.size(); ++i)
        {
            const auto bx = block_sizes[i].first;
            const auto by = block_sizes[i].second;

            t1 = Clock::now();
            processor.sobel_cuda_global(gray.data, cuda3.data, gray.cols, gray.rows, 3, bx, by, &timing);
            t2 = Clock::now();
            print_breakdown("CUDA Global", 3, bx, by, timing);
            std::cout << "CUDA Global 3x3 [" << bx << "x" << by << "]: "
                      << std::chrono::duration<double, std::milli>(t2 - t1).count()
                      << " ms\n";

            t1 = Clock::now();
            processor.sobel_cuda_shared(gray.data, cuda3.data, gray.cols, gray.rows, 3, bx, by, &timing);
            t2 = Clock::now();
            print_breakdown("CUDA Shared", 3, bx, by, timing);
            std::cout << "CUDA Shared 3x3 [" << bx << "x" << by << "]: "
                      << std::chrono::duration<double, std::milli>(t2 - t1).count()
                      << " ms\n";
        }

        cv::imshow("GPU 3x3", cuda3);
        cv::imshow("GPU 5x5", cuda5);
        cv::waitKey(0);
        return 0;
    }

    int run_video_benchmark(const std::string &path)
    {
        cv::VideoCapture cap(path);
        if (!cap.isOpened())
        {
            std::cout << "Could not open video\n";
            return -1;
        }

        std::cout << "Video benchmark: " << path << '\n';
        std::cout << "Frames reported below are averages over the full video.\n";

        SobelProcessor processor;
        processor.cuda_setup();

        VideoStats global3 = benchmark_video_gpu(cap, processor,
                                                 [&](const cv::Mat &gray, cv::Mat &out, CudaTimingBreakdown &timing)
                                                 {
                                                     processor.sobel_cuda_global(gray.data, out.data, gray.cols, gray.rows, 3, 16, 16, &timing);
                                                 });
        print_video_stats("CUDA Global 3x3 video", global3);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        VideoStats shared3 = benchmark_video_gpu(cap, processor,
                                                 [&](const cv::Mat &gray, cv::Mat &out, CudaTimingBreakdown &timing)
                                                 {
                                                     processor.sobel_cuda_shared(gray.data, out.data, gray.cols, gray.rows, 3, 16, 16, &timing);
                                                 });
        print_video_stats("CUDA Shared 3x3 video", shared3);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        VideoStats global5 = benchmark_video_gpu(cap, processor,
                                                 [&](const cv::Mat &gray, cv::Mat &out, CudaTimingBreakdown &timing)
                                                 {
                                                     processor.sobel_cuda_global(gray.data, out.data, gray.cols, gray.rows, 5, 16, 16, &timing);
                                                 });
        print_video_stats("CUDA Global 5x5 video", global5);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        VideoStats shared5 = benchmark_video_gpu(cap, processor,
                                                 [&](const cv::Mat &gray, cv::Mat &out, CudaTimingBreakdown &timing)
                                                 {
                                                     processor.sobel_cuda_shared(gray.data, out.data, gray.cols, gray.rows, 5, 16, 16, &timing);
                                                 });
        print_video_stats("CUDA Shared 5x5 video", shared5);

        const std::array<VideoVariant, 4> variants = {{
            {"CUDA Global 3x3 video", 3, false, global3},
            {"CUDA Shared 3x3 video", 3, true, shared3},
            {"CUDA Global 5x5 video", 5, false, global5},
            {"CUDA Shared 5x5 video", 5, true, shared5},
        }};

        const VideoVariant *best_variant = &variants[0];
        for (const auto &variant : variants)
        {
            if (average_total_ms(variant.stats) < average_total_ms(best_variant->stats))
                best_variant = &variant;
        }

        preview_video_gpu(path, best_variant->label,
                          [&](const cv::Mat &gray, cv::Mat &out, CudaTimingBreakdown &timing)
                          {
                              if (best_variant->use_shared)
                                  processor.sobel_cuda_shared(gray.data, out.data, gray.cols, gray.rows, best_variant->kernel_size, 16, 16, &timing);
                              else
                                  processor.sobel_cuda_global(gray.data, out.data, gray.cols, gray.rows, best_variant->kernel_size, 16, 16, &timing);
                          });

        return 0;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <image-or-video-path>\n";
        return -1;
    }

    const std::string input_path = argv[1];
    if (!is_image_path(input_path) && is_video_path(input_path))
        return run_video_benchmark(input_path);

    return run_image_benchmark(input_path);
}
