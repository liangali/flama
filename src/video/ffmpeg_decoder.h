//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#pragma once

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

// Initialize FFmpeg network
void init_ffmpeg();

// Find the video stream index in a format context
int find_video_stream(AVFormatContext *format_context);

// Open a hardware decoder (e.g., D3D11VA)
AVCodecContext *open_decoder(AVFormatContext *format_context,
                             int stream_index,
                             const char *hw_device_type);

// Open a software decoder with multi-threading support
AVCodecContext *open_decoder_sw(AVFormatContext *format_context, int stream_index);

// Cleanup FFmpeg resources
void cleanup(AVFormatContext *format_context, AVCodecContext *codec_context);

// Prefer D3D11 hardware pixel format during decoding when offered
enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);
