#include <opencv2/opencv.hpp>
#include <array>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include "sobel.hpp"

// Configuration
#define PREVIEW_VIDEO 1

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

    void print_stats(const char *label, int k, int bx, int by, const CudaTiming_t &timing)
    {
        std::cout << "  " << label << " breakdown (K=" << k << ", block=" << bx << "x" << by << "): "
                  << "H2D=" << timing.h2d_ms << " ms, ";
        if (timing.grayscale_ms > 0.0f)
            std::cout << "Gray=" << timing.grayscale_ms << " ms, ";
        std::cout
            << "Kernel=" << timing.kernel_ms << " ms, "
            << "D2H=" << timing.d2h_ms << " ms\n";
    }

    struct VideoStats_t
    {
        double total_ms = 0.0;
        double read_decode_ms = 0.0;
        double cpu_grayscale_ms = 0.0;
        double host_overhead_ms = 0.0;
        double gpu_total_ms = 0.0;
        double h2d_ms = 0.0;
        double gpu_grayscale_ms = 0.0;
        double kernel_ms = 0.0;
        double d2h_ms = 0.0;
        double gpu_overhead_ms = 0.0;
        int frames = 0;
    };

    struct VideoTestTypes_t
    {
        const char *label = "";
        int kernel_size = 3;
        bool use_shared = false;
        bool use_gpu_grayscale = false;
        VideoStats_t stats;
    };

    template <typename Func>
    VideoStats_t benchmark_video_gpu(cv::VideoCapture &cap, SobelProcessor &processor, Func run_gpu)
    {
        VideoStats_t stats;
        cv::Mat frame;

        while (true)
        {
            auto total_start = Clock::now();
            auto read_start = total_start;
            if (!cap.read(frame))
                break;
            auto read_end = Clock::now();

            auto grayscale_start = Clock::now();
            cv::Mat gray = to_grayscale(frame);
            auto grayscale_end = Clock::now();
            cv::Mat out(gray.size(), gray.type());
            CudaTiming_t timing;

            auto gpu_start = Clock::now();
            run_gpu(gray, out, timing);
            auto gpu_end = Clock::now();

            const double total_ms = std::chrono::duration<double, std::milli>(gpu_end - total_start).count();
            const double read_decode_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
            const double grayscale_ms = std::chrono::duration<double, std::milli>(grayscale_end - grayscale_start).count();
            const double gpu_total_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
            const double gpu_breakdown_ms = timing.h2d_ms + timing.grayscale_ms + timing.kernel_ms + timing.d2h_ms;
            const double gpu_overhead_ms = std::max(0.0, gpu_total_ms - gpu_breakdown_ms);
            const double host_overhead_ms = std::max(0.0, total_ms - read_decode_ms - grayscale_ms - gpu_total_ms);

            stats.total_ms += total_ms;
            stats.read_decode_ms += read_decode_ms;
            stats.cpu_grayscale_ms += grayscale_ms;
            stats.host_overhead_ms += host_overhead_ms;
            stats.gpu_total_ms += gpu_total_ms;
            stats.h2d_ms += timing.h2d_ms;
            stats.gpu_grayscale_ms += timing.grayscale_ms;
            stats.kernel_ms += timing.kernel_ms;
            stats.d2h_ms += timing.d2h_ms;
            stats.gpu_overhead_ms += gpu_overhead_ms;
            stats.frames += 1;
        }

        return stats;
    }

    template <typename Func>
    VideoStats_t benchmark_video_gpu_bgr(cv::VideoCapture &cap, SobelProcessor &processor, Func run_gpu)
    {
        VideoStats_t stats;
        cv::Mat frame;

        while (true)
        {
            auto total_start = Clock::now();
            auto read_start = total_start;
            if (!cap.read(frame))
                break;
            auto read_end = Clock::now();

            cv::Mat out(frame.rows, frame.cols, CV_8UC1);
            CudaTiming_t timing;

            auto gpu_start = Clock::now();
            run_gpu(frame, out, timing);
            auto gpu_end = Clock::now();

            const double total_ms = std::chrono::duration<double, std::milli>(gpu_end - total_start).count();
            const double read_decode_ms = std::chrono::duration<double, std::milli>(read_end - read_start).count();
            const double gpu_total_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
            const double gpu_breakdown_ms = timing.h2d_ms + timing.grayscale_ms + timing.kernel_ms + timing.d2h_ms;
            const double gpu_overhead_ms = std::max(0.0, gpu_total_ms - gpu_breakdown_ms);
            const double host_overhead_ms = std::max(0.0, total_ms - read_decode_ms - gpu_total_ms);

            stats.total_ms += total_ms;
            stats.read_decode_ms += read_decode_ms;
            stats.cpu_grayscale_ms += 0.0;
            stats.host_overhead_ms += host_overhead_ms;
            stats.gpu_total_ms += gpu_total_ms;
            stats.h2d_ms += timing.h2d_ms;
            stats.gpu_grayscale_ms += timing.grayscale_ms;
            stats.kernel_ms += timing.kernel_ms;
            stats.d2h_ms += timing.d2h_ms;
            stats.gpu_overhead_ms += gpu_overhead_ms;
            stats.frames += 1;
        }

        return stats;
    }

    void print_video_stats(const char *label, const VideoStats_t &stats)
    {
        if (stats.frames == 0)
            return;

        double avg_total = stats.total_ms / stats.frames;
        double avg_read_decode = stats.read_decode_ms / stats.frames;
        double avg_cpu_grayscale = stats.cpu_grayscale_ms / stats.frames;
        double avg_host_overhead = stats.host_overhead_ms / stats.frames;
        double avg_gpu_total = stats.gpu_total_ms / stats.frames;
        double avg_h2d = stats.h2d_ms / stats.frames;
        double avg_gpu_grayscale = stats.gpu_grayscale_ms / stats.frames;
        double avg_kernel = stats.kernel_ms / stats.frames;
        double avg_d2h = stats.d2h_ms / stats.frames;
        double avg_gpu_overhead = stats.gpu_overhead_ms / stats.frames;
        double avg_sum = avg_read_decode + avg_cpu_grayscale + avg_host_overhead + avg_h2d + avg_gpu_grayscale + avg_kernel + avg_d2h + avg_gpu_overhead;
        double fps = 1000.0 / avg_total;

        std::cout << label << ": "
                  << stats.frames << " frames, "
                  << "avg total=" << avg_total << " ms, "
                  << "FPS=" << fps << '\n';
        std::cout << "  avg host stages: "
                  << "Read/Decode=" << avg_read_decode << " ms, "
                  << "CPU Gray=" << avg_cpu_grayscale << " ms, "
                  << "Host overhead=" << avg_host_overhead << " ms\n";
        std::cout << "  avg GPU stages: "
                  << "GPU total=" << avg_gpu_total << " ms, "
                  << "H2D=" << avg_h2d << " ms, "
                  << "GPU Gray=" << avg_gpu_grayscale << " ms, "
                  << "Kernel=" << avg_kernel << " ms, "
                  << "D2H=" << avg_d2h << " ms, "
                  << "GPU overhead=" << avg_gpu_overhead << " ms\n";
        std::cout << "  avg summed components=" << avg_sum << " ms\n";
    }

    double average_total_ms(const VideoStats_t &stats)
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
            cv::Mat out(frame.rows, frame.cols, CV_8UC1);
            CudaTiming_t timing;
            run_gpu(frame, out, timing);

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
        CudaTiming_t timing;

        t1 = Clock::now();
        processor.sobel_cuda_global(gray.data, cuda3.data, gray.cols, gray.rows, 3, 16, 16, &timing);
        t2 = Clock::now();
        print_stats("CUDA Global", 3, 16, 16, timing);
        std::cout << "CUDA Global 3x3: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared(gray.data, cuda3.data, gray.cols, gray.rows, 3, 16, 16, &timing);
        t2 = Clock::now();
        print_stats("CUDA Shared", 3, 16, 16, timing);
        std::cout << "CUDA Shared 3x3: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        cv::Mat cuda5(gray.size(), gray.type());

        t1 = Clock::now();
        processor.sobel_cuda_global(gray.data, cuda5.data, gray.cols, gray.rows, 5, 16, 16, &timing);
        t2 = Clock::now();
        print_stats("CUDA Global", 5, 16, 16, timing);
        std::cout << "CUDA Global 5x5: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared(gray.data, cuda5.data, gray.cols, gray.rows, 5, 16, 16, &timing);
        t2 = Clock::now();
        print_stats("CUDA Shared", 5, 16, 16, timing);
        std::cout << "CUDA Shared 5x5: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        cv::Mat cuda3_gpu_gray(gray.size(), gray.type());
        cv::Mat cuda5_gpu_gray(gray.size(), gray.type());

        std::cout << "\nGPU grayscale + Sobel\n";
        t1 = Clock::now();
        processor.sobel_cuda_global_bgr(img.data, cuda3_gpu_gray.data, img.cols, img.rows, img.channels(), 3, 16, 16, &timing);
        t2 = Clock::now();
        print_stats("CUDA Global + GPU Gray", 3, 16, 16, timing);
        std::cout << "CUDA Global 3x3 (GPU gray): " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared_bgr(img.data, cuda3_gpu_gray.data, img.cols, img.rows, img.channels(), 3, 16, 16, &timing);
        t2 = Clock::now();
        print_stats("CUDA Shared + GPU Gray", 3, 16, 16, timing);
        std::cout << "CUDA Shared 3x3 (GPU gray): " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_global_bgr(img.data, cuda5_gpu_gray.data, img.cols, img.rows, img.channels(), 5, 16, 16, &timing);
        t2 = Clock::now();
        print_stats("CUDA Global + GPU Gray", 5, 16, 16, timing);
        std::cout << "CUDA Global 5x5 (GPU gray): " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared_bgr(img.data, cuda5_gpu_gray.data, img.cols, img.rows, img.channels(), 5, 16, 16, &timing);
        t2 = Clock::now();
        print_stats("CUDA Shared + GPU Gray", 5, 16, 16, timing);
        std::cout << "CUDA Shared 5x5 (GPU gray): " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";

        std::cout << "\nBlock trials (chrono total per call)\n";
        for (int i = 0; i < block_sizes.size(); ++i)
        {
            const auto bx = block_sizes[i].first;
            const auto by = block_sizes[i].second;

            t1 = Clock::now();
            processor.sobel_cuda_global(gray.data, cuda3.data, gray.cols, gray.rows, 3, bx, by, &timing);
            t2 = Clock::now();
            print_stats("CUDA Global", 3, bx, by, timing);
            std::cout << "CUDA Global 3x3 [" << bx << "x" << by << "]: "
                      << std::chrono::duration<double, std::milli>(t2 - t1).count()
                      << " ms\n";

            t1 = Clock::now();
            processor.sobel_cuda_shared(gray.data, cuda3.data, gray.cols, gray.rows, 3, bx, by, &timing);
            t2 = Clock::now();
            print_stats("CUDA Shared", 3, bx, by, timing);
            std::cout << "CUDA Shared 3x3 [" << bx << "x" << by << "]: "
                      << std::chrono::duration<double, std::milli>(t2 - t1).count()
                      << " ms\n";
        }

        cv::imshow("GPU 3x3", cuda3);
        cv::imshow("GPU 5x5", cuda5);
        cv::imshow("GPU Gray 3x3", cuda3_gpu_gray);
        cv::imshow("GPU Gray 5x5", cuda5_gpu_gray);
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

        VideoStats_t global3 = benchmark_video_gpu(cap, processor,
                                                   [&](const cv::Mat &gray, cv::Mat &out, CudaTiming_t &timing)
                                                   {
                                                       processor.sobel_cuda_global(gray.data, out.data, gray.cols, gray.rows, 3, 16, 16, &timing);
                                                   });
        print_video_stats("CUDA Global 3x3 video", global3);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset Video Position
        VideoStats_t shared3 = benchmark_video_gpu(cap, processor,
                                                   [&](const cv::Mat &gray, cv::Mat &out, CudaTiming_t &timing)
                                                   {
                                                       processor.sobel_cuda_shared(gray.data, out.data, gray.cols, gray.rows, 3, 16, 16, &timing);
                                                   });
        print_video_stats("CUDA Shared 3x3 video", shared3);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset Video Position
        VideoStats_t global5 = benchmark_video_gpu(cap, processor,
                                                   [&](const cv::Mat &gray, cv::Mat &out, CudaTiming_t &timing)
                                                   {
                                                       processor.sobel_cuda_global(gray.data, out.data, gray.cols, gray.rows, 5, 16, 16, &timing);
                                                   });
        print_video_stats("CUDA Global 5x5 video", global5);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset Video Position
        VideoStats_t shared5 = benchmark_video_gpu(cap, processor,
                                                   [&](const cv::Mat &gray, cv::Mat &out, CudaTiming_t &timing)
                                                   {
                                                       processor.sobel_cuda_shared(gray.data, out.data, gray.cols, gray.rows, 5, 16, 16, &timing);
                                                   });
        print_video_stats("CUDA Shared 5x5 video", shared5);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset Video Position
        VideoStats_t global3_gpu_gray = benchmark_video_gpu_bgr(cap, processor,
                                                                [&](const cv::Mat &frame, cv::Mat &out, CudaTiming_t &timing)
                                                                {
                                                                    processor.sobel_cuda_global_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), 3, 16, 16, &timing);
                                                                });
        print_video_stats("CUDA Global 3x3 video (GPU gray)", global3_gpu_gray);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset Video Position
        VideoStats_t shared3_gpu_gray = benchmark_video_gpu_bgr(cap, processor,
                                                                [&](const cv::Mat &frame, cv::Mat &out, CudaTiming_t &timing)
                                                                {
                                                                    processor.sobel_cuda_shared_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), 3, 16, 16, &timing);
                                                                });
        print_video_stats("CUDA Shared 3x3 video (GPU gray)", shared3_gpu_gray);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset Video Position
        VideoStats_t global5_gpu_gray = benchmark_video_gpu_bgr(cap, processor,
                                                                [&](const cv::Mat &frame, cv::Mat &out, CudaTiming_t &timing)
                                                                {
                                                                    processor.sobel_cuda_global_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), 5, 16, 16, &timing);
                                                                });
        print_video_stats("CUDA Global 5x5 video (GPU gray)", global5_gpu_gray);

        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Reset Video Position
        VideoStats_t shared5_gpu_gray = benchmark_video_gpu_bgr(cap, processor,
                                                                [&](const cv::Mat &frame, cv::Mat &out, CudaTiming_t &timing)
                                                                {
                                                                    processor.sobel_cuda_shared_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), 5, 16, 16, &timing);
                                                                });
        print_video_stats("CUDA Shared 5x5 video (GPU gray)", shared5_gpu_gray);

        const std::array<VideoTestTypes_t, 8> uut = {{
            {"CUDA Global 3x3 video", 3, false, false, global3},
            {"CUDA Shared 3x3 video", 3, true, false, shared3},
            {"CUDA Global 5x5 video", 5, false, false, global5},
            {"CUDA Shared 5x5 video", 5, true, false, shared5},
            {"CUDA Global 3x3 video (GPU gray)", 3, false, true, global3_gpu_gray},
            {"CUDA Shared 3x3 video (GPU gray)", 3, true, true, shared3_gpu_gray},
            {"CUDA Global 5x5 video (GPU gray)", 5, false, true, global5_gpu_gray},
            {"CUDA Shared 5x5 video (GPU gray)", 5, true, true, shared5_gpu_gray},
        }};

        const VideoTestTypes_t *best = &uut[0];
        for (const auto &test : uut)
        {
            if (average_total_ms(test.stats) < average_total_ms(best->stats))
                best = &test;
        }

        std::cout << "Best Variant (avg total ms): " << best->label << std::endl;

#if PREVIEW_VIDEO == 1
        preview_video_gpu(path, best->label,
                          [&](const cv::Mat &frame, cv::Mat &out, CudaTiming_t &timing)
                          {
                              if (best->use_gpu_grayscale)
                              {
                                  if (best->use_shared)
                                      processor.sobel_cuda_shared_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), best->kernel_size, 16, 16, &timing);
                                  else
                                      processor.sobel_cuda_global_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), best->kernel_size, 16, 16, &timing);
                              }
                              else if (best->use_shared)
                              {
                                  cv::Mat gray = to_grayscale(frame);
                                  processor.sobel_cuda_shared(gray.data, out.data, gray.cols, gray.rows, best->kernel_size, 16, 16, &timing);
                              }
                              else
                              {
                                  cv::Mat gray = to_grayscale(frame);
                                  processor.sobel_cuda_global(gray.data, out.data, gray.cols, gray.rows, best->kernel_size, 16, 16, &timing);
                              }
                          });
#endif
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
