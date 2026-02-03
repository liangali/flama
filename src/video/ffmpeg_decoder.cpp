//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "ffmpeg_decoder.h"
#include "../utils/debug.h"
#include "../utils/util.h"

#include <iostream>
#include <string>
#include <thread>

extern "C"
{
#include <libswscale/swscale.h>
}

void init_ffmpeg()
{
    avformat_network_init();
}

int find_video_stream(AVFormatContext *format_context)
{
    for (unsigned int i = 0; i < format_context->nb_streams; i++)
    {
        if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p = pix_fmts;
    bool has_d3d11 = false;
    std::string offered;
    while (*p != AV_PIX_FMT_NONE)
    {
        const char *name = av_get_pix_fmt_name(*p);
        offered += (offered.empty() ? "" : ", ");
        offered += (name ? name : "<unknown>");
        if (*p == AV_PIX_FMT_D3D11)
            has_d3d11 = true;
        ++p;
    }
    DBG_LOGF("[HW] get_format offered: %s", offered.c_str());

    p = pix_fmts; // reset pointer to start
    while (*p != AV_PIX_FMT_NONE)
    {
        if (*p == AV_PIX_FMT_D3D11)
        {
            DBG_LOG("[HW] Selecting AV_PIX_FMT_D3D11 for hardware frames");
            return *p;
        }
        ++p;
    }

    // Fallback to default selection and log
    DBG_LOG("[HW] D3D11 pixel format not offered; falling back to first offered software format");
    // Choose the first format offered by FFmpeg (standard behavior)
    return pix_fmts && pix_fmts[0] != AV_PIX_FMT_NONE ? pix_fmts[0] : AV_PIX_FMT_NONE;
}

AVCodecContext *open_decoder(AVFormatContext *format_context,
                             int stream_index,
                             const char *hw_device_type)
{
    AVCodecParameters *codec_params = format_context->streams[stream_index]->codecpar;
    const AVCodec *codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec)
    {
        std::cerr << "Failed to find decoder for codec_id=" << codec_params->codec_id << std::endl;
        return nullptr;
    }

    AVCodecContext *codec_context = avcodec_alloc_context3(codec);
    if (!codec_context)
    {
        std::cerr << "Failed to allocate codec context." << std::endl;
        return nullptr;
    }

    if (avcodec_parameters_to_context(codec_context, codec_params) < 0)
    {
        std::cerr << "Failed to copy codec parameters to codec context." << std::endl;
        avcodec_free_context(&codec_context);
        return nullptr;
    }

    AVBufferRef *hw_device_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_device_ctx,
                               av_hwdevice_find_type_by_name(hw_device_type),
                               nullptr,
                               nullptr,
                               0) < 0)
    {
        std::cerr << "Failed to create hardware device context." << std::endl;
        avcodec_free_context(&codec_context);
        return nullptr;
    }
    codec_context->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    if (avcodec_open2(codec_context, codec, nullptr) < 0)
    {
        std::cerr << "Failed to open codec." << std::endl;
        av_buffer_unref(&hw_device_ctx);
        avcodec_free_context(&codec_context);
        return nullptr;
    }

    // --- Diagnostics: decoder + HW device info ---
    {
        const char *codec_name = codec ? codec->name : "<null>";
        const char *profile_name = avcodec_profile_name(codec_context->codec_id, codec_context->profile);
        const char *device_type_name = nullptr;
        if (codec_context->hw_device_ctx)
        {
            AVHWDeviceContext *devctx = (AVHWDeviceContext *)codec_context->hw_device_ctx->data;
            device_type_name = av_hwdevice_get_type_name(devctx ? devctx->type : AV_HWDEVICE_TYPE_NONE);
        }
        DBG_LOGF("[HW] Decoder opened: codec=%s profile=%s width=%d height=%d hw_device=%s get_format=%s",
                 codec_name,
                 profile_name ? profile_name : "<unknown>",
                 codec_context->width,
                 codec_context->height,
                 device_type_name ? device_type_name : "<none>",
                 (codec_context->get_format ? "set" : "null"));

        // List hardware configs offered by this codec (for visibility)
        int i = 0;
        while (true)
        {
            const AVCodecHWConfig *cfg = avcodec_get_hw_config(codec, i);
            if (!cfg)
                break;
            const char *pf = av_get_pix_fmt_name(cfg->pix_fmt);
            const char *dt = av_hwdevice_get_type_name(cfg->device_type);
            DBG_LOGF("[HW] codec hw_config[%d]: device=%s pix_fmt=%s methods=0x%x",
                     i, dt ? dt : "<none>", pf ? pf : "<unknown>", cfg->methods);
            ++i;
        }
    }

    av_buffer_unref(&hw_device_ctx);
    return codec_context;
}

AVCodecContext *open_decoder_sw(AVFormatContext *format_context, int stream_index)
{
    AVCodecParameters *codec_params = format_context->streams[stream_index]->codecpar;
    const AVCodec *codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec)
    {
        std::cerr << "Failed to find decoder for codec_id=" << codec_params->codec_id << std::endl;
        return nullptr;
    }
    AVCodecContext *codec_context = avcodec_alloc_context3(codec);
    if (!codec_context)
    {
        std::cerr << "[SW] Failed to allocate codec context." << std::endl;
        return nullptr;
    }
    if (avcodec_parameters_to_context(codec_context, codec_params) < 0)
    {
        std::cerr << "[SW] Failed to copy codec parameters." << std::endl;
        avcodec_free_context(&codec_context);
        return nullptr;
    }

    if ((codec_context->codec_id == AV_CODEC_ID_HEVC || codec_context->codec_id == AV_CODEC_ID_H264) &&
        codec_context->thread_count == 0)
    {
        codec_context->thread_count = std::thread::hardware_concurrency();
        if (codec_context->thread_count == 0)
            codec_context->thread_count = 1;
        DBG_LOG(std::string("[SW] Enable multithread decode, threads=") + std::to_string(codec_context->thread_count));
    }
    if (avcodec_open2(codec_context, codec, nullptr) < 0)
    {
        std::cerr << "[SW] Failed to open software decoder." << std::endl;
        avcodec_free_context(&codec_context);
        return nullptr;
    }
    DBG_LOG(std::string("[SW] Software decoder opened: ") + codec->name + " threads=" + std::to_string(codec_context->thread_count));
    return codec_context;
}

void cleanup(AVFormatContext *format_context, AVCodecContext *codec_context)
{
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);
    avformat_network_deinit();
}
