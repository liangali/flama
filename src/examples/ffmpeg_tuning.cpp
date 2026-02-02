#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <windows.h>
// FFmpeg library headers
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/time.h>
#include <libavutil/pixfmt.h>
}

// Macro: Check FFmpeg function return values
#define FF_CHECK(call, msg) \
    if ((call) < 0) { \
        char errbuf[AV_ERROR_MAX_STRING_SIZE]; \
        av_strerror(ret, errbuf, sizeof(errbuf)); \
        std::cerr << "FFmpeg Error: " << msg << " failed with code " << ret << " (" << errbuf << ")\n"; \
        return ret; \
    }

// Global hardware context reference
AVBufferRef* hw_device_ctx = nullptr;

/**
 * @brief Callback function to select AV_PIX_FMT_D3D11 pixel format
 */
static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    const enum AVPixelFormat* p;
    for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
        if (*p == AV_PIX_FMT_D3D11) {
            std::cout << "Selected AV_PIX_FMT_D3D11 as hardware pixel format.\n";
            return *p;
        }
    }
    std::cerr << "ERROR: Decoder does not support AV_PIX_FMT_D3D11 pixel format.\n";
    return AV_PIX_FMT_NONE;
}

/**
 * @brief Initialize D3D11VA hardware device context
 */
static int init_d3d11va_context() {
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("d3d11va");
    if (type == AV_HWDEVICE_TYPE_NONE) {
        std::cerr << "ERROR: D3D11VA hardware device type is not available. Check FFmpeg compilation and system environment.\n";
        return -1;
    }

    int ret = av_hwdevice_ctx_create(&hw_device_ctx, type, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "ERROR: Failed to create D3D11VA hardware context: " << ret << "\n";
        return ret;
    }
    std::cout << "Successfully created D3D11VA hardware context.\n";
    return 0;
}


// -----------------------------------------------------------------
// Core functions: FFmpeg initialization, decoding, and performance analysis
// -----------------------------------------------------------------

/**
 * @brief Core decoding and performance profiling function
 * @param filename Path to video file
 * @param use_hardware Whether to use hardware decoding
 */
int profile_ffmpeg_decoding(const std::string& filename, bool use_hardware) {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* dec_ctx = nullptr;
    const AVCodec* decoder = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    int video_stream_idx = -1;
    int ret = 0;

    std::vector<double> decode_times_ms;
    long long total_start_us = av_gettime();
    long long decode_start_us = 0;
    long long decode_end_us = 0;
    long long total_end_us = 0;
    int total_frames = 0;

    // Step 1: Initialize hardware context if hardware decoding is requested
    if (use_hardware) {
        if (init_d3d11va_context() < 0) {
            return -1; // Hardware initialization failed
        }
    }

    // Step 2: Open input file
    ret = avformat_open_input(&fmt_ctx, filename.c_str(), nullptr, nullptr);
    FF_CHECK(ret, "avformat_open_input");

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    FF_CHECK(ret, "avformat_find_stream_info");

    ret = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (ret < 0) {
        std::cerr << "ERROR: No video stream found\n";
        goto end;
    }
    video_stream_idx = ret;

    // Step 3: Initialize decoder context
    dec_ctx = avcodec_alloc_context3(decoder);
    if (!dec_ctx) {
        ret = AVERROR(ENOMEM);
        goto end;
    }
    avcodec_parameters_to_context(dec_ctx, fmt_ctx->streams[video_stream_idx]->codecpar);

    // Step 4: Configure hardware acceleration (hardware mode only)
    if (use_hardware) {
        if (!hw_device_ctx) {
            // This should not happen, but safe in error handling
            std::cerr << "FATAL ERROR: Hardware context not initialized.\n";
            ret = -1;
            goto end;
        }
        dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        dec_ctx->get_format = get_hw_format;
        std::cout << "Using D3D11VA hardware accelerated decoding.\n";
    }
    else {
        std::cout << "Using CPU software decoding.\n";
    }

    // Step 5: Open decoder
    ret = avcodec_open2(dec_ctx, decoder, nullptr);
    FF_CHECK(ret, "avcodec_open2");

    // Step 6: Allocate packet and frame
    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!pkt || !frame) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    // Step 7: Decoding and timing loop
    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        printf("[verbose] av_read_frame\n");
        if (pkt->stream_index == video_stream_idx) {
            printf("[verbose] pkt->stream_index == video_stream_idx\n");
            decode_start_us = av_gettime();

            ret = avcodec_send_packet(dec_ctx, pkt);
            if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                // Ignore AVERROR(EAGAIN) and AVERROR_EOF
                FF_CHECK(ret, "avcodec_send_packet");
            }

            while (ret >= 0) {
                printf("[verbose] avcodec_receive_frame\n");
                ret = avcodec_receive_frame(dec_ctx, frame);

                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                FF_CHECK(ret, "avcodec_receive_frame");

                decode_end_us = av_gettime();

                double time_taken_ms = (decode_end_us - decode_start_us) / 1000.0;
                decode_times_ms.push_back(time_taken_ms);
                std::cout << "decode time for frame : " << time_taken_ms << " ms" << std::endl;
                decode_start_us = av_gettime();
            }
        }
        printf("[verbose] av_packet_unref\n");
        av_packet_unref(pkt);
    }
    printf("[verbose] avcodec_send_packet\n");
    // 8. Drain the decoder
    avcodec_send_packet(dec_ctx, nullptr);
    while (avcodec_receive_frame(dec_ctx, frame) >= 0) {
        printf("[verbose] avcodec_receive_frame\n");
        decode_end_us = av_gettime();
        double time_taken_ms = (decode_end_us - decode_start_us) / 1000.0;
        decode_times_ms.push_back(time_taken_ms);
        decode_start_us = av_gettime();
    }

    // 9. 
    total_end_us = av_gettime();
    total_frames = decode_times_ms.size();

    if (total_frames > 0) {
        double total_decode_time_ms = std::accumulate(decode_times_ms.begin(), decode_times_ms.end(), 0.0);
        double avg_decode_time_ms = total_decode_time_ms / total_frames;
        double total_time_s = (total_end_us - total_start_us) / 1000000.0;
        double decode_fps = total_frames / total_time_s;
        double min_time = *std::min_element(decode_times_ms.begin(), decode_times_ms.end());
        double max_time = *std::max_element(decode_times_ms.begin(), decode_times_ms.end());

        std::string mode = use_hardware ? "D3D11VA 硬件" : "CPU 软件";

        std::cout << "\n========================================\n";
        std::cout << "✨ " << mode << "解码性能分析结果:\n";
        std::cout << "========================================\n";
        std::cout << "总解码帧数: " << total_frames << " 帧\n";
        std::cout << "总运行时间: " << total_time_s << " 秒\n";
        std::cout << "🚀 实际解码帧率: " << decode_fps << " FPS\n";
        std::cout << "⏱️ 平均单帧解码耗时: " << avg_decode_time_ms << " 毫秒\n";
        std::cout << "  (最小耗时: " << min_time << " ms, 最大耗时: " << max_time << " ms)\n";
    }
    else {
        std::cout << "⚠️ 未成功解码任何帧。\n";
    }

end:
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
        hw_device_ctx = nullptr;
    }

    return 0;
}

#include <iostream>
#include <string>
#include <fstream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
#include <libavutil/pixdesc.h>
}

#define FF_CHECK(ret, msg, cleanup_label) \
    if ((ret) < 0) { \
        char errbuf[AV_ERROR_MAX_STRING_SIZE]; \
        av_strerror((ret), errbuf, sizeof(errbuf)); \
        std::cerr << "FFmpeg Error: " << msg << " failed with code " << (ret) << " (" << errbuf << ")\n"; \
        goto cleanup_label; \
    }

AVBufferRef* g_hw_device_ctx = nullptr;

int create_hw_device_context() {
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name("d3d11va");
    if (type == AV_HWDEVICE_TYPE_NONE) {
        std::cerr << "❌ D3D11VA 设备不可用。\n";
        return -1;
    }

    int ret = av_hwdevice_ctx_create(&g_hw_device_ctx, type, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "❌ 创建 AVHWDeviceContext 失败。\n";
        return ret;
    }
    return 0;
}

AVBufferRef* init_hw_frames_context(AVCodecContext* dec_ctx) {
    int ret = 0;
    AVBufferRef* frames_ctx_ref = nullptr;
    AVHWFramesContext* frames_ctx = nullptr;

    frames_ctx_ref = av_hwframe_ctx_alloc(g_hw_device_ctx);
    if (!frames_ctx_ref) return nullptr;

    frames_ctx = (AVHWFramesContext*)frames_ctx_ref->data;
    frames_ctx->format = AV_PIX_FMT_D3D11;
    frames_ctx->width = dec_ctx->width;
    frames_ctx->height = dec_ctx->height;
    frames_ctx->initial_pool_size = 10; 

    ret = av_hwframe_ctx_init(frames_ctx_ref);
    if (ret < 0) {
        av_buffer_unref(&frames_ctx_ref);
        return nullptr;
    }

    return frames_ctx_ref;
}

int decode_and_process(const std::string& filename) {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* dec_ctx = nullptr;
    const AVCodec* decoder = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    AVBufferRef* hw_frames_ctx_ref = nullptr;
    int video_stream_idx = -1;
    int ret = 0;
    int decoded_frames = 0;

    if (create_hw_device_context() < 0) {
        goto end;
    }

    ret = avformat_open_input(&fmt_ctx, filename.c_str(), nullptr, nullptr);
    FF_CHECK(ret, "avformat_open_input", end);

    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    FF_CHECK(ret, "avformat_find_stream_info", end);

    ret = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (ret < 0) {
        std::cerr << "错误：未找到视频流。\n";
        goto end;
    }
    video_stream_idx = ret;

    dec_ctx = avcodec_alloc_context3(decoder);
    if (!dec_ctx) {
        ret = AVERROR(ENOMEM);
        goto end;
    }
    avcodec_parameters_to_context(dec_ctx, fmt_ctx->streams[video_stream_idx]->codecpar);

    dec_ctx->get_format = get_hw_format;

    ret = avcodec_open2(dec_ctx, decoder, nullptr);
    FF_CHECK(ret, "avcodec_open2", end);

    hw_frames_ctx_ref = init_hw_frames_context(dec_ctx);
    if (!hw_frames_ctx_ref) {
        std::cerr << " AVHWFramesContext init failed, exit!\n";
        goto end;
    }
   // dec_ctx->hw_frames_ctx = hw_frames_ctx_ref;

    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!pkt || !frame) {
        ret = AVERROR(ENOMEM);
        goto end;
    }


    while (av_read_frame(fmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == video_stream_idx) {
            ret = avcodec_send_packet(dec_ctx, pkt);
            while (ret >= 0) {
                ret = avcodec_receive_frame(dec_ctx, frame);

                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                FF_CHECK(ret, "avcodec_receive_frame", end);

                if (frame->format == AV_PIX_FMT_D3D11) {

                }
                else {
                    std::cout << "  [SW] Warning, fallback to SW decoder, Format = " << av_get_pix_fmt_name((AVPixelFormat)frame->format) << "\n";
                }

                av_frame_unref(frame);
            }
        }
        av_packet_unref(pkt);
    }

    std::cout << "\n finished decode " << decoded_frames << " frames\n";

end:
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&dec_ctx);
    avformat_close_input(&fmt_ctx);

    if (g_hw_device_ctx) {
        av_buffer_unref(&g_hw_device_ctx);
    }

    return ret;
}