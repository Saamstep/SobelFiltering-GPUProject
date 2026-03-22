#include <opencv2/opencv.hpp>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include "sobel.hpp"

// Configuration
#define PREVIEW_VIDEO 1
#define VIDEO_BENCH_PIPELINED 0
#define VIDEO_PIPELINE_RING_SIZE 3

namespace
{
    using Clock = std::chrono::high_resolution_clock;
    constexpr int kDefaultBlockX = 16;
    constexpr int kDefaultBlockY = 16;

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

    double elapsed_ms(Clock::time_point start, Clock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    void ensure_contiguous(cv::Mat &mat)
    {
        if (!mat.isContinuous())
            mat = mat.clone();
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
        std::cout << "Kernel=" << timing.kernel_ms << " ms, "
                  << "D2H=" << timing.d2h_ms << " ms\n";
    }

    struct VideoStats_t
    {
        double total_ms = 0.0;
        double latency_total_ms = 0.0;
        double throughput_wall_ms = 0.0;
        double read_decode_ms = 0.0;
        double cpu_grayscale_ms = 0.0;
        double cpu_sobel_ms = 0.0;
        double host_staging_ms = 0.0;
        double pipeline_wait_ms = 0.0;
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

    std::array<VideoTestTypes_t, 8> make_video_tests()
    {
        return {{
            {"CUDA Global 3x3 video", 3, false, false, {}},
            {"CUDA Shared 3x3 video", 3, true, false, {}},
            {"CUDA Global 5x5 video", 5, false, false, {}},
            {"CUDA Shared 5x5 video", 5, true, false, {}},
            {"CUDA Global 3x3 video (GPU gray)", 3, false, true, {}},
            {"CUDA Shared 3x3 video (GPU gray)", 3, true, true, {}},
            {"CUDA Global 5x5 video (GPU gray)", 5, false, true, {}},
            {"CUDA Shared 5x5 video (GPU gray)", 5, true, true, {}},
        }};
    }

    template <typename Func>
    VideoStats_t benchmark_video_gpu(cv::VideoCapture &cap, Func run_gpu)
    {
        VideoStats_t stats;
        cv::Mat frame;

        while (true)
        {
            const auto total_start = Clock::now();
            const auto read_start = total_start;
            if (!cap.read(frame))
                break;
            const auto read_end = Clock::now();

            const auto grayscale_start = Clock::now();
            cv::Mat gray = to_grayscale(frame);
            ensure_contiguous(gray);
            const auto grayscale_end = Clock::now();

            cv::Mat out(gray.size(), gray.type());
            CudaTiming_t timing;

            const auto gpu_start = Clock::now();
            run_gpu(gray, out, timing);
            const auto gpu_end = Clock::now();

            const double total_ms = elapsed_ms(total_start, gpu_end);
            const double read_decode_ms = elapsed_ms(read_start, read_end);
            const double grayscale_ms = elapsed_ms(grayscale_start, grayscale_end);
            const double gpu_total_ms = elapsed_ms(gpu_start, gpu_end);
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

    template <int K>
    VideoStats_t benchmark_video_cpu(cv::VideoCapture &cap)
    {
        VideoStats_t stats;
        cv::Mat frame;
        SobelProcessor processor;

        while (true)
        {
            const auto total_start = Clock::now();
            const auto read_start = total_start;
            if (!cap.read(frame))
                break;
            const auto read_end = Clock::now();

            const auto grayscale_start = Clock::now();
            cv::Mat gray = to_grayscale(frame);
            ensure_contiguous(gray);
            const auto grayscale_end = Clock::now();

            cv::Mat out(gray.size(), gray.type());
            const auto sobel_start = Clock::now();
            processor.sobel_cpu<K>(gray.data, out.data, gray.cols, gray.rows);
            const auto sobel_end = Clock::now();

            const double total_ms = elapsed_ms(total_start, sobel_end);
            const double read_decode_ms = elapsed_ms(read_start, read_end);
            const double grayscale_ms = elapsed_ms(grayscale_start, grayscale_end);
            const double cpu_sobel_ms = elapsed_ms(sobel_start, sobel_end);
            const double host_overhead_ms = std::max(0.0, total_ms - read_decode_ms - grayscale_ms - cpu_sobel_ms);

            stats.total_ms += total_ms;
            stats.read_decode_ms += read_decode_ms;
            stats.cpu_grayscale_ms += grayscale_ms;
            stats.cpu_sobel_ms += cpu_sobel_ms;
            stats.host_overhead_ms += host_overhead_ms;
            stats.frames += 1;
        }

        return stats;
    }

    template <typename Func>
    VideoStats_t benchmark_video_gpu_bgr(cv::VideoCapture &cap, Func run_gpu)
    {
        VideoStats_t stats;
        cv::Mat frame;

        while (true)
        {
            const auto total_start = Clock::now();
            const auto read_start = total_start;
            if (!cap.read(frame))
                break;
            const auto read_end = Clock::now();

            ensure_contiguous(frame);
            cv::Mat out(frame.rows, frame.cols, CV_8UC1);
            CudaTiming_t timing;

            const auto gpu_start = Clock::now();
            run_gpu(frame, out, timing);
            const auto gpu_end = Clock::now();

            const double total_ms = elapsed_ms(total_start, gpu_end);
            const double read_decode_ms = elapsed_ms(read_start, read_end);
            const double gpu_total_ms = elapsed_ms(gpu_start, gpu_end);
            const double gpu_breakdown_ms = timing.h2d_ms + timing.grayscale_ms + timing.kernel_ms + timing.d2h_ms;
            const double gpu_overhead_ms = std::max(0.0, gpu_total_ms - gpu_breakdown_ms);
            const double host_overhead_ms = std::max(0.0, total_ms - read_decode_ms - gpu_total_ms);

            stats.total_ms += total_ms;
            stats.read_decode_ms += read_decode_ms;
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

    void accumulate_pipeline_stats(VideoStats_t &stats, const CudaPipelineFrameTiming_t &timing)
    {
        if (timing.frame_index < 0)
            return;

        stats.latency_total_ms += timing.total_latency_ms;
        stats.read_decode_ms += timing.read_decode_ms;
        stats.cpu_grayscale_ms += timing.cpu_grayscale_ms;
        stats.host_staging_ms += timing.host_staging_ms;
        stats.pipeline_wait_ms += timing.pipeline_wait_ms;
        stats.gpu_total_ms += timing.h2d_ms + timing.gpu_grayscale_ms + timing.kernel_ms + timing.d2h_ms + timing.gpu_overhead_ms;
        stats.h2d_ms += timing.h2d_ms;
        stats.gpu_grayscale_ms += timing.gpu_grayscale_ms;
        stats.kernel_ms += timing.kernel_ms;
        stats.d2h_ms += timing.d2h_ms;
        stats.gpu_overhead_ms += timing.gpu_overhead_ms;
        stats.frames += 1;
    }

    VideoStats_t benchmark_video_gpu_pipelined(cv::VideoCapture &cap, SobelProcessor &processor, const VideoTestTypes_t &test)
    {
        VideoStats_t stats;
        CudaPipelineContext_t context;
        cv::Mat frame;
        int frame_index = 0;
        bool pipeline_ready = false;
        const auto wall_start = Clock::now();

        while (true)
        {
            const auto read_start = Clock::now();
            if (!cap.read(frame))
                break;
            const auto read_end = Clock::now();

            ensure_contiguous(frame);

            if (!pipeline_ready)
            {
                CudaPipelineConfig_t config;
                config.width = frame.cols;
                config.height = frame.rows;
                config.channels = frame.channels();
                config.kernel_size = test.kernel_size;
                config.block_x = kDefaultBlockX;
                config.block_y = kDefaultBlockY;
                config.use_shared = test.use_shared;
                config.use_gpu_grayscale = test.use_gpu_grayscale;
                config.ring_depth = VIDEO_PIPELINE_RING_SIZE;
                processor.cuda_pipeline_init(config, context);
                pipeline_ready = true;
            }

            CudaPipelineFrameTiming_t frame_timing;
            frame_timing.frame_index = frame_index;
            frame_timing.read_decode_ms = elapsed_ms(read_start, read_end);

            cv::Mat gray;
            if (!test.use_gpu_grayscale)
            {
                const auto gray_start = Clock::now();
                gray = to_grayscale(frame);
                ensure_contiguous(gray);
                const auto gray_end = Clock::now();
                frame_timing.cpu_grayscale_ms = elapsed_ms(gray_start, gray_end);
            }

            const int slot_index = frame_index % context.config.ring_depth;
            CudaPipelineSlot_t &slot = context.slots[slot_index];

            const auto wait_start = Clock::now();
            if (slot.in_flight)
            {
                CudaPipelineFrameTiming_t completed;
                processor.cuda_pipeline_finalize(context, slot_index, completed);
                accumulate_pipeline_stats(stats, completed);
            }
            const auto wait_end = Clock::now();
            frame_timing.pipeline_wait_ms = elapsed_ms(wait_start, wait_end);

            const auto staging_start = Clock::now();
            if (test.use_gpu_grayscale)
            {
                const size_t color_bytes = static_cast<size_t>(frame.cols) * frame.rows * frame.channels() * sizeof(unsigned char);
                std::memcpy(slot.h_color, frame.data, color_bytes);
            }
            else
            {
                const size_t gray_bytes = static_cast<size_t>(gray.cols) * gray.rows * sizeof(unsigned char);
                std::memcpy(slot.h_gray, gray.data, gray_bytes);
            }
            const auto staging_end = Clock::now();
            frame_timing.host_staging_ms = elapsed_ms(staging_start, staging_end);

            processor.cuda_pipeline_enqueue(context, slot_index, frame_timing);
            frame_index += 1;
        }

        if (pipeline_ready)
        {
            for (int slot_index = 0; slot_index < context.config.ring_depth; ++slot_index)
            {
                if (!context.slots[slot_index].in_flight)
                    continue;

                CudaPipelineFrameTiming_t completed;
                processor.cuda_pipeline_finalize(context, slot_index, completed);
                accumulate_pipeline_stats(stats, completed);
            }
        }

        const auto wall_end = Clock::now();
        stats.throughput_wall_ms = elapsed_ms(wall_start, wall_end);
        processor.cuda_pipeline_destroy(context);
        return stats;
    }

    void print_video_stats(const char *label, const VideoStats_t &stats)
    {
        if (stats.frames == 0)
            return;

        const double avg_total = stats.total_ms / stats.frames;
        const double avg_read_decode = stats.read_decode_ms / stats.frames;
        const double avg_cpu_grayscale = stats.cpu_grayscale_ms / stats.frames;
        const double avg_host_overhead = stats.host_overhead_ms / stats.frames;
        const double avg_gpu_total = stats.gpu_total_ms / stats.frames;
        const double avg_h2d = stats.h2d_ms / stats.frames;
        const double avg_gpu_grayscale = stats.gpu_grayscale_ms / stats.frames;
        const double avg_kernel = stats.kernel_ms / stats.frames;
        const double avg_d2h = stats.d2h_ms / stats.frames;
        const double avg_gpu_overhead = stats.gpu_overhead_ms / stats.frames;
        const double avg_sum = avg_read_decode + avg_cpu_grayscale + avg_host_overhead + avg_h2d + avg_gpu_grayscale + avg_kernel + avg_d2h + avg_gpu_overhead;
        const double fps = 1000.0 / avg_total;

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

    void print_video_stats_cpu(const char *label, const VideoStats_t &stats)
    {
        if (stats.frames == 0)
            return;

        const double avg_total = stats.total_ms / stats.frames;
        const double avg_read_decode = stats.read_decode_ms / stats.frames;
        const double avg_cpu_grayscale = stats.cpu_grayscale_ms / stats.frames;
        const double avg_cpu_sobel = stats.cpu_sobel_ms / stats.frames;
        const double avg_host_overhead = stats.host_overhead_ms / stats.frames;
        const double avg_sum = avg_read_decode + avg_cpu_grayscale + avg_cpu_sobel + avg_host_overhead;
        const double fps = 1000.0 / avg_total;

        std::cout << label << ": "
                  << stats.frames << " frames, "
                  << "avg total=" << avg_total << " ms, "
                  << "FPS=" << fps << '\n';
        std::cout << "  avg CPU stages: "
                  << "Read/Decode=" << avg_read_decode << " ms, "
                  << "CPU Gray=" << avg_cpu_grayscale << " ms, "
                  << "CPU Sobel=" << avg_cpu_sobel << " ms, "
                  << "Host overhead=" << avg_host_overhead << " ms\n";
        std::cout << "  avg summed components=" << avg_sum << " ms\n";
    }

    void print_video_stats_pipelined(const char *label, const VideoStats_t &stats)
    {
        if (stats.frames == 0)
            return;

        const double avg_throughput_frame_ms = stats.throughput_wall_ms / stats.frames;
        const double throughput_fps = stats.throughput_wall_ms > 0.0 ? (1000.0 * stats.frames) / stats.throughput_wall_ms : 0.0;
        const double avg_latency = stats.latency_total_ms / stats.frames;
        const double avg_read_decode = stats.read_decode_ms / stats.frames;
        const double avg_cpu_grayscale = stats.cpu_grayscale_ms / stats.frames;
        const double avg_host_staging = stats.host_staging_ms / stats.frames;
        const double avg_pipeline_wait = stats.pipeline_wait_ms / stats.frames;
        const double avg_h2d = stats.h2d_ms / stats.frames;
        const double avg_gpu_grayscale = stats.gpu_grayscale_ms / stats.frames;
        const double avg_kernel = stats.kernel_ms / stats.frames;
        const double avg_d2h = stats.d2h_ms / stats.frames;
        const double avg_gpu_overhead = stats.gpu_overhead_ms / stats.frames;
        const double avg_sum = avg_read_decode + avg_cpu_grayscale + avg_host_staging + avg_pipeline_wait + avg_h2d + avg_gpu_grayscale + avg_kernel + avg_d2h + avg_gpu_overhead;

        std::cout << label << " [pipelined]: " << stats.frames << " frames\n";
        std::cout << "  throughput: wall=" << stats.throughput_wall_ms << " ms, "
                  << "avg frame interval=" << avg_throughput_frame_ms << " ms, "
                  << "FPS=" << throughput_fps << '\n';
        std::cout << "  avg frame latency=" << avg_latency << " ms\n";
        std::cout << "  avg latency stages: "
                  << "Read/Decode=" << avg_read_decode << " ms, "
                  << "CPU Gray=" << avg_cpu_grayscale << " ms, "
                  << "Host staging=" << avg_host_staging << " ms, "
                  << "Pipeline wait=" << avg_pipeline_wait << " ms, "
                  << "H2D=" << avg_h2d << " ms, "
                  << "GPU Gray=" << avg_gpu_grayscale << " ms, "
                  << "Kernel=" << avg_kernel << " ms, "
                  << "D2H=" << avg_d2h << " ms, "
                  << "GPU overhead=" << avg_gpu_overhead << " ms\n";
        std::cout << "  avg summed latency components=" << avg_sum << " ms\n";
    }

    double average_total_ms(const VideoStats_t &stats)
    {
        if (stats.frames == 0)
            return std::numeric_limits<double>::infinity();
        return stats.total_ms / stats.frames;
    }

    double average_pipeline_frame_ms(const VideoStats_t &stats)
    {
        if (stats.frames == 0 || stats.throughput_wall_ms <= 0.0)
            return std::numeric_limits<double>::infinity();
        return stats.throughput_wall_ms / stats.frames;
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
            ensure_contiguous(frame);
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

    void preview_best_video_variant(const std::string &path, SobelProcessor &processor, const VideoTestTypes_t &best)
    {
#if PREVIEW_VIDEO == 1
        preview_video_gpu(path, best.label,
                          [&](const cv::Mat &frame, cv::Mat &out, CudaTiming_t &timing)
                          {
                              if (best.use_gpu_grayscale)
                              {
                                  if (best.use_shared)
                                      processor.sobel_cuda_shared_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), best.kernel_size, kDefaultBlockX, kDefaultBlockY, &timing);
                                  else
                                      processor.sobel_cuda_global_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), best.kernel_size, kDefaultBlockX, kDefaultBlockY, &timing);
                                  return;
                              }

                              cv::Mat gray = to_grayscale(frame);
                              ensure_contiguous(gray);
                              if (best.use_shared)
                                  processor.sobel_cuda_shared(gray.data, out.data, gray.cols, gray.rows, best.kernel_size, kDefaultBlockX, kDefaultBlockY, &timing);
                              else
                                  processor.sobel_cuda_global(gray.data, out.data, gray.cols, gray.rows, best.kernel_size, kDefaultBlockX, kDefaultBlockY, &timing);
                          });
#endif
    }

    VideoStats_t benchmark_video_variant_serial(cv::VideoCapture &cap, SobelProcessor &processor, const VideoTestTypes_t &test)
    {
        if (test.use_gpu_grayscale)
        {
            return benchmark_video_gpu_bgr(cap,
                                           [&](const cv::Mat &frame, cv::Mat &out, CudaTiming_t &timing)
                                           {
                                               if (test.use_shared)
                                                   processor.sobel_cuda_shared_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), test.kernel_size, kDefaultBlockX, kDefaultBlockY, &timing);
                                               else
                                                   processor.sobel_cuda_global_bgr(frame.data, out.data, frame.cols, frame.rows, frame.channels(), test.kernel_size, kDefaultBlockX, kDefaultBlockY, &timing);
                                           });
        }

        return benchmark_video_gpu(cap,
                                   [&](const cv::Mat &gray, cv::Mat &out, CudaTiming_t &timing)
                                   {
                                       if (test.use_shared)
                                           processor.sobel_cuda_shared(gray.data, out.data, gray.cols, gray.rows, test.kernel_size, kDefaultBlockX, kDefaultBlockY, &timing);
                                       else
                                           processor.sobel_cuda_global(gray.data, out.data, gray.cols, gray.rows, test.kernel_size, kDefaultBlockX, kDefaultBlockY, &timing);
                                   });
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
        ensure_contiguous(gray);
        ensure_contiguous(img);
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
        std::cout << "CPU 3x3: " << elapsed_ms(t1, t2) << " ms\n";

        cv::Mat cpu5(gray.size(), gray.type());
        t1 = Clock::now();
        processor.sobel_cpu<5>(gray.data, cpu5.data, gray.cols, gray.rows);
        t2 = Clock::now();
        std::cout << "CPU 5x5: " << elapsed_ms(t1, t2) << " ms\n";

        processor.cuda_setup();
        std::cout << "CUDA warm-up completed before timing.\n";

        cv::Mat cuda3(gray.size(), gray.type());
        CudaTiming_t timing;

        t1 = Clock::now();
        processor.sobel_cuda_global(gray.data, cuda3.data, gray.cols, gray.rows, 3, kDefaultBlockX, kDefaultBlockY, &timing);
        t2 = Clock::now();
        print_stats("CUDA Global", 3, kDefaultBlockX, kDefaultBlockY, timing);
        std::cout << "CUDA Global 3x3: " << elapsed_ms(t1, t2) << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared(gray.data, cuda3.data, gray.cols, gray.rows, 3, kDefaultBlockX, kDefaultBlockY, &timing);
        t2 = Clock::now();
        print_stats("CUDA Shared", 3, kDefaultBlockX, kDefaultBlockY, timing);
        std::cout << "CUDA Shared 3x3: " << elapsed_ms(t1, t2) << " ms\n";

        cv::Mat cuda5(gray.size(), gray.type());

        t1 = Clock::now();
        processor.sobel_cuda_global(gray.data, cuda5.data, gray.cols, gray.rows, 5, kDefaultBlockX, kDefaultBlockY, &timing);
        t2 = Clock::now();
        print_stats("CUDA Global", 5, kDefaultBlockX, kDefaultBlockY, timing);
        std::cout << "CUDA Global 5x5: " << elapsed_ms(t1, t2) << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared(gray.data, cuda5.data, gray.cols, gray.rows, 5, kDefaultBlockX, kDefaultBlockY, &timing);
        t2 = Clock::now();
        print_stats("CUDA Shared", 5, kDefaultBlockX, kDefaultBlockY, timing);
        std::cout << "CUDA Shared 5x5: " << elapsed_ms(t1, t2) << " ms\n";

        cv::Mat cuda3_gpu_gray(gray.size(), gray.type());
        cv::Mat cuda5_gpu_gray(gray.size(), gray.type());

        std::cout << "\nGPU grayscale + Sobel\n";
        t1 = Clock::now();
        processor.sobel_cuda_global_bgr(img.data, cuda3_gpu_gray.data, img.cols, img.rows, img.channels(), 3, kDefaultBlockX, kDefaultBlockY, &timing);
        t2 = Clock::now();
        print_stats("CUDA Global + GPU Gray", 3, kDefaultBlockX, kDefaultBlockY, timing);
        std::cout << "CUDA Global 3x3 (GPU gray): " << elapsed_ms(t1, t2) << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared_bgr(img.data, cuda3_gpu_gray.data, img.cols, img.rows, img.channels(), 3, kDefaultBlockX, kDefaultBlockY, &timing);
        t2 = Clock::now();
        print_stats("CUDA Shared + GPU Gray", 3, kDefaultBlockX, kDefaultBlockY, timing);
        std::cout << "CUDA Shared 3x3 (GPU gray): " << elapsed_ms(t1, t2) << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_global_bgr(img.data, cuda5_gpu_gray.data, img.cols, img.rows, img.channels(), 5, kDefaultBlockX, kDefaultBlockY, &timing);
        t2 = Clock::now();
        print_stats("CUDA Global + GPU Gray", 5, kDefaultBlockX, kDefaultBlockY, timing);
        std::cout << "CUDA Global 5x5 (GPU gray): " << elapsed_ms(t1, t2) << " ms\n";

        t1 = Clock::now();
        processor.sobel_cuda_shared_bgr(img.data, cuda5_gpu_gray.data, img.cols, img.rows, img.channels(), 5, kDefaultBlockX, kDefaultBlockY, &timing);
        t2 = Clock::now();
        print_stats("CUDA Shared + GPU Gray", 5, kDefaultBlockX, kDefaultBlockY, timing);
        std::cout << "CUDA Shared 5x5 (GPU gray): " << elapsed_ms(t1, t2) << " ms\n";

        std::cout << "\nBlock trials (chrono total per call)\n";
        for (int i = 0; i < static_cast<int>(block_sizes.size()); ++i)
        {
            const int bx = block_sizes[i].first;
            const int by = block_sizes[i].second;

            t1 = Clock::now();
            processor.sobel_cuda_global(gray.data, cuda3.data, gray.cols, gray.rows, 3, bx, by, &timing);
            t2 = Clock::now();
            print_stats("CUDA Global", 3, bx, by, timing);
            std::cout << "CUDA Global 3x3 [" << bx << "x" << by << "]: " << elapsed_ms(t1, t2) << " ms\n";

            t1 = Clock::now();
            processor.sobel_cuda_shared(gray.data, cuda3.data, gray.cols, gray.rows, 3, bx, by, &timing);
            t2 = Clock::now();
            print_stats("CUDA Shared", 3, bx, by, timing);
            std::cout << "CUDA Shared 3x3 [" << bx << "x" << by << "]: " << elapsed_ms(t1, t2) << " ms\n";
        }

        cv::imshow("CPU 5x5", cpu5);
        // cv::imshow("GPU 3x3", cuda3);
        // cv::imshow("GPU 5x5", cuda5);
        // cv::imshow("GPU Gray 3x3", cuda3_gpu_gray);
        // cv::imshow("GPU Gray 5x5", cuda5_gpu_gray);
        cv::waitKey(0);
        return 0;
    }

    int run_video_benchmark_serial(const std::string &path)
    {
        cv::VideoCapture cap(path);
        if (!cap.isOpened())
        {
            std::cout << "Could not open video\n";
            return -1;
        }

        std::cout << "Video benchmark (serial): " << path << '\n';
        std::cout << "Frames reported below are averages over the full video.\n";

        print_video_stats_cpu("CPU 3x3 video", benchmark_video_cpu<3>(cap));
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        print_video_stats_cpu("CPU 5x5 video", benchmark_video_cpu<5>(cap));
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        SobelProcessor processor;
        processor.cuda_setup();
        auto tests = make_video_tests();

        for (VideoTestTypes_t &test : tests)
        {
            test.stats = benchmark_video_variant_serial(cap, processor, test);
            print_video_stats(test.label, test.stats);
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        }

        const VideoTestTypes_t *best = &tests[0];
        for (const auto &test : tests)
        {
            if (average_total_ms(test.stats) < average_total_ms(best->stats))
                best = &test;
        }

        std::cout << "Best Variant (avg total ms): " << best->label << '\n';
        preview_best_video_variant(path, processor, *best);
        return 0;
    }

    int run_video_benchmark_pipelined(const std::string &path)
    {
        cv::VideoCapture cap(path);
        if (!cap.isOpened())
        {
            std::cout << "Could not open video\n";
            return -1;
        }

        std::cout << "Video benchmark (pipelined): " << path << '\n';
        std::cout << "Frames reported below show throughput separately from per-frame latency.\n";

        std::cout << "CPU baseline below is serial over the same video frames.\n";
        print_video_stats_cpu("CPU 3x3 video", benchmark_video_cpu<3>(cap));
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        print_video_stats_cpu("CPU 5x5 video", benchmark_video_cpu<5>(cap));
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        SobelProcessor processor;
        processor.cuda_setup();
        auto tests = make_video_tests();

        for (VideoTestTypes_t &test : tests)
        {
            test.stats = benchmark_video_gpu_pipelined(cap, processor, test);
            print_video_stats_pipelined(test.label, test.stats);
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        }

        const VideoTestTypes_t *best = &tests[0];
        for (const auto &test : tests)
        {
            if (average_pipeline_frame_ms(test.stats) < average_pipeline_frame_ms(best->stats))
                best = &test;
        }

        std::cout << "Best Variant (throughput avg frame interval): " << best->label << '\n';
        preview_best_video_variant(path, processor, *best);
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
    {
#if VIDEO_BENCH_PIPELINED == 1
        return run_video_benchmark_pipelined(input_path);
#else
        return run_video_benchmark_serial(input_path);
#endif
    }

    return run_image_benchmark(input_path);
}
