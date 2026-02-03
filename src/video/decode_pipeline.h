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
}

// Software decode path: decode + scale BGRA + inference
void decode_frames_sw(AVFormatContext *format_context, AVCodecContext *codec_context, int stream_index);

// Hardware decode path: decode + VPP scale + inference (D3D11VA)
void decode_frames(AVFormatContext *format_context, AVCodecContext *codec_context, int stream_index);

// Timing accumulators
extern uint64_t g_sw_inference_total_us;
extern uint64_t g_sw_pipeline_total_us;
extern uint64_t g_hw_inference_total_us;
extern uint64_t g_hw_pipeline_total_us;
