//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "decode_pipeline.h"
#include "frame_selector.h"
#include "segment_timing.h"
#include "vpp.h"
#include "batch/batch_state.h"
#include "inference/cb_processing.h"
#include "inference/vlm_chat.h"
#include "utils/util.h"
#include "utils/debug.h"
#include "utils/profiling.h"
#include "device/texture_resource_pool.h"

#include <iostream>
#include <chrono>

extern "C"
{
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#ifdef _WIN32
#include <libavutil/hwcontext_d3d11va.h>
#endif

#define MAX_FRAME_QUEUE 128

// Timing accumulators are defined in batch_state.cpp
extern uint64_t g_sw_inference_total_us;
extern uint64_t g_sw_pipeline_total_us;
extern uint64_t g_hw_inference_total_us;
extern uint64_t g_hw_pipeline_total_us;

void decode_frames_sw(AVFormatContext *format_context, AVCodecContext *codec_context, int stream_index)
{
    auto sw_total_start_tp = std::chrono::steady_clock::now();
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    AVFrame *frameBGRA = av_frame_alloc();
    if (!packet || !frame || !frameBGRA)
    {
        std::cerr << "[SW] Allocation failure for decoding structures." << std::endl;
        return;
    }
    struct SwsContext *swsCtx = nullptr;
    int frameCounter = 0;
    int target_width = g_commonConfig.vpp_down_width;
    int target_height = g_commonConfig.vpp_down_height;
    auto &frameProfiler = prof::FrameProfiler::Get();
    AVPixelFormat lastSrcFmt = AV_PIX_FMT_NONE;
    int lastW = 0, lastH = 0;

    FrameSelector selector(g_fsConfig);
    AVRational time_base = format_context->streams[stream_index]->time_base;

    auto pipeline_start_us = std::chrono::steady_clock::now();
    while (av_read_frame(format_context, packet) >= 0)
    {
        if (packet->stream_index != stream_index)
        {
            av_packet_unref(packet);
            continue;
        }

        frameProfiler.BeginFrame(frameCounter);
        frameProfiler.MarkStageBegin(prof::Stage::Pipeline);
        frameProfiler.MarkStageBegin(prof::Stage::Decode);

        if (avcodec_send_packet(codec_context, packet) < 0)
        {
            std::cerr << "[SW] Error sending packet." << std::endl;
            av_packet_unref(packet);
            frameProfiler.MarkStageEnd(prof::Stage::Decode);
            if (g_batchConfig.new_batch_mode)
            {
                const auto &fr = frameProfiler.Current();
                const auto &sr = fr.stages[(int)prof::Stage::Decode];
                if (sr.end_us >= sr.start_us)
                    g_batchState.decode_total_us += (sr.end_us - sr.start_us);
                g_batchState.windowDecoded++;
            }
            frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
            frameProfiler.EndFrameAndWrite();
            continue;
        }
        av_packet_unref(packet);

        while (true)
        {
            int ret = avcodec_receive_frame(codec_context, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0)
            {
                std::cerr << "[SW] avcodec_receive_frame error: " << ret << std::endl;
                break;
            }
            ++frameCounter;
            frameProfiler.MarkStageEnd(prof::Stage::Decode);
            if (g_batchConfig.new_batch_mode)
            {
                const auto &fr = frameProfiler.Current();
                const auto &sr = fr.stages[(int)prof::Stage::Decode];
                if (sr.end_us >= sr.start_us)
                    g_batchState.decode_total_us += (sr.end_us - sr.start_us);
                g_batchState.windowDecoded++;
            }

            AVFrame *selectedFrame = nullptr;
            bool key = (frame->flags & AV_FRAME_FLAG_KEY) != 0;
            bool sceneCut = false;
            if (!selector.AcceptDecodedFrame(frame, time_base, selectedFrame))
            {
                frameProfiler.SetSelectionFlags(false, key, sceneCut);
                prof::Summary::Get().RecordFrame(false, key, sceneCut);
                frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
                frameProfiler.EndFrameAndWrite();
                frameProfiler.MarkStageBegin(prof::Stage::Decode);
                continue;
            }
            AVFrame *srcForScale = selectedFrame;
            frameProfiler.SetSelectionFlags(true, key, sceneCut);
            prof::Summary::Get().RecordFrame(true, key, sceneCut);
            if (g_batchConfig.new_batch_mode)
            {
                double pts_sec = GetFramePtsSeconds(selectedFrame, time_base);
                UpdateBatchSegmentTiming(pts_sec);
            }

            if (frameCounter == 1)
            {
                DBG_LOG(std::string("[SW] First selected frame fmt=") + av_get_pix_fmt_name((AVPixelFormat)srcForScale->format) + " size=" + std::to_string(srcForScale->width) + "x" + std::to_string(srcForScale->height));
            }
            if (srcForScale->format != lastSrcFmt || srcForScale->width != lastW || srcForScale->height != lastH)
            {
                const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((AVPixelFormat)srcForScale->format);
                int bitDepth = desc ? desc->comp[0].depth : 0;
                DBG_LOG(std::string("[SW] Format/size change (selected): ") + (lastSrcFmt == AV_PIX_FMT_NONE ? "(initial)" : av_get_pix_fmt_name(lastSrcFmt)) + " -> " + av_get_pix_fmt_name((AVPixelFormat)srcForScale->format) + " (bitdepth=" + std::to_string(bitDepth) + ")" + ", " + std::to_string(lastW) + "x" + std::to_string(lastH) + " -> " + std::to_string(srcForScale->width) + "x" + std::to_string(srcForScale->height));
                lastSrcFmt = (AVPixelFormat)srcForScale->format;
                lastW = srcForScale->width;
                lastH = srcForScale->height;
                if (swsCtx)
                {
                    sws_freeContext(swsCtx);
                    swsCtx = nullptr;
                }
                av_frame_unref(frameBGRA);
                frameBGRA->format = AV_PIX_FMT_BGRA;
                frameBGRA->width = target_width;
                frameBGRA->height = target_height;
                if (av_frame_get_buffer(frameBGRA, 32) < 0)
                {
                    std::cerr << "[SW] Failed to alloc BGRA frame buffer." << std::endl;
                    av_frame_free(&selectedFrame);
                    break;
                }
            }

            if (g_batchConfig.new_batch_mode)
            {
                auto tScale0 = std::chrono::steady_clock::now();
                AVFrame *scaled = ScaleFrameSW(srcForScale, target_width, target_height, swsCtx);
                auto tScale1 = std::chrono::steady_clock::now();
                if (scaled)
                {
                    g_batchState.cached_scaled.push_back(scaled);
                    g_batchState.scale_total_us += std::chrono::duration_cast<std::chrono::microseconds>(tScale1 - tScale0).count();
                }
                av_frame_free(&selectedFrame);
                g_batchState.windowSelected++;
                if ((int)g_batchState.cached_scaled.size() >= g_batchConfig.max_cache || ((int)g_batchState.cached_scaled.size() % g_batchConfig.batch_trigger) == 0)
                {
                    RunBatchSW();
                    auto pipeline_end_us = std::chrono::steady_clock::now();
                    g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
                    prof::BatchAggregator::Get().RecordBatch(g_batchState.windowDecoded, g_batchState.windowSelected, g_batchState.decode_total_us, g_batchState.scale_total_us, g_batchState.tensor_total_us, g_batchState.inference_us, g_batchState.pipeline_us, g_batchState.prompt, g_batchState.result);
                    RecordJsonSegmentFromBatch(g_batchState, g_batchState.result);
                    DBG_LOGF("[Batch] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu pipeline_us=%llu", (unsigned long long)g_batchState.batchIndex, g_batchState.cached_scaled.size(), (unsigned long long)g_batchState.windowDecoded, (unsigned long long)g_batchState.windowSelected, (unsigned long long)g_batchState.scale_total_us, (unsigned long long)g_batchState.tensor_total_us, (unsigned long long)g_batchState.inference_us, (unsigned long long)g_batchState.pipeline_us);
                    g_batchState.reset();
                    pipeline_start_us = std::chrono::steady_clock::now();
                }
            }
            else
            {
                frameProfiler.MarkStageBegin(prof::Stage::Scale);
                if (!swsCtx)
                {
                    swsCtx = sws_getContext(srcForScale->width, srcForScale->height, (AVPixelFormat)srcForScale->format,
                                            target_width, target_height, AV_PIX_FMT_BGRA,
                                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                    if (!swsCtx)
                    {
                        std::cerr << "[SW] Failed to create swsContext (srcFmt="
                                  << av_get_pix_fmt_name((AVPixelFormat)srcForScale->format) << ")" << std::endl;
                        frameProfiler.MarkStageEnd(prof::Stage::Scale);
                        av_frame_free(&selectedFrame);
                        break;
                    }
                }
                if (sws_scale(swsCtx, srcForScale->data, srcForScale->linesize, 0, srcForScale->height,
                              frameBGRA->data, frameBGRA->linesize) <= 0)
                {
                    std::cerr << "[SW] sws_scale failed." << std::endl;
                    frameProfiler.MarkStageEnd(prof::Stage::Scale);
                    av_frame_free(&selectedFrame);
                    continue;
                }
                frameProfiler.MarkStageEnd(prof::Stage::Scale);
#ifdef _WIN32
                SceneUnderstandSW(m_pD3D11Device, m_pD3D11DeviceContext, frameBGRA);
#endif
                av_frame_free(&selectedFrame);
            }

            frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
            frameProfiler.EndFrameAndWrite();
            frameProfiler.MarkStageBegin(prof::Stage::Decode);
        }
    }

    frameProfiler.BeginFrame(frameCounter);
    frameProfiler.MarkStageBegin(prof::Stage::Pipeline);
    frameProfiler.MarkStageBegin(prof::Stage::Decode);
    avcodec_send_packet(codec_context, nullptr);
    while (true)
    {
        int ret = avcodec_receive_frame(codec_context, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        if (ret < 0)
            break;
        ++frameCounter;
        frameProfiler.MarkStageEnd(prof::Stage::Decode);
        if (g_batchConfig.new_batch_mode)
        {
            const auto &fr = frameProfiler.Current();
            const auto &sr = fr.stages[(int)prof::Stage::Decode];
            if (sr.end_us >= sr.start_us)
                g_batchState.decode_total_us += (sr.end_us - sr.start_us);
            g_batchState.windowDecoded++;
        }
        AVFrame *selectedFrame = nullptr;
        bool key = (frame->flags & AV_FRAME_FLAG_KEY) != 0;
        bool sceneCut = false;
        if (!selector.AcceptDecodedFrame(frame, format_context->streams[stream_index]->time_base, selectedFrame))
        {
            frameProfiler.SetSelectionFlags(false, key, sceneCut);
            prof::Summary::Get().RecordFrame(false, key, sceneCut);
            frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
            frameProfiler.EndFrameAndWrite();
            frameProfiler.MarkStageBegin(prof::Stage::Decode);
            continue;
        }
        frameProfiler.SetSelectionFlags(true, key, sceneCut);
        prof::Summary::Get().RecordFrame(true, key, sceneCut);
        if (g_batchConfig.new_batch_mode)
        {
            double pts_sec = GetFramePtsSeconds(selectedFrame, format_context->streams[stream_index]->time_base);
            UpdateBatchSegmentTiming(pts_sec);
        }
        AVFrame *flushScaleSrc = selectedFrame;
        if (g_batchConfig.new_batch_mode)
        {
            auto tScale0 = std::chrono::steady_clock::now();
            AVFrame *scaled = ScaleFrameSW(flushScaleSrc, target_width, target_height, swsCtx);
            auto tScale1 = std::chrono::steady_clock::now();
            if (scaled)
            {
                g_batchState.cached_scaled.push_back(scaled);
                g_batchState.scale_total_us += std::chrono::duration_cast<std::chrono::microseconds>(tScale1 - tScale0).count();
            }
            av_frame_free(&selectedFrame);
            g_batchState.windowSelected++;
        }
        else
        {
            frameProfiler.MarkStageBegin(prof::Stage::Scale);
            if (!swsCtx)
            {
                swsCtx = sws_getContext(flushScaleSrc->width, flushScaleSrc->height, (AVPixelFormat)flushScaleSrc->format,
                                        target_width, target_height, AV_PIX_FMT_BGRA,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!swsCtx)
                {
                    std::cerr << "[SW] Failed to create swsContext in flush." << std::endl;
                    frameProfiler.MarkStageEnd(prof::Stage::Scale);
                    av_frame_free(&selectedFrame);
                    break;
                }
            }
            if (sws_scale(swsCtx, flushScaleSrc->data, flushScaleSrc->linesize, 0, flushScaleSrc->height,
                          frameBGRA->data, frameBGRA->linesize) <= 0)
            {
                std::cerr << "[SW] sws_scale failed (flush)." << std::endl;
                frameProfiler.MarkStageEnd(prof::Stage::Scale);
                av_frame_free(&selectedFrame);
                continue;
            }
            frameProfiler.MarkStageEnd(prof::Stage::Scale);
            av_frame_free(&selectedFrame);
        }
        frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
        frameProfiler.EndFrameAndWrite();
        frameProfiler.MarkStageBegin(prof::Stage::Decode);
    }

    if (swsCtx)
        sws_freeContext(swsCtx);
    if (g_batchConfig.new_batch_mode && !g_batchState.cached_scaled.empty() && g_batchConfig.flush_partial)
    {
        RunBatchSW();
        auto pipeline_end_us = std::chrono::steady_clock::now();
        g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
        prof::BatchAggregator::Get().RecordBatch(g_batchState.windowDecoded, g_batchState.windowSelected, g_batchState.decode_total_us, g_batchState.scale_total_us, g_batchState.tensor_total_us, g_batchState.inference_us, g_batchState.pipeline_us, g_batchState.prompt, g_batchState.result);
        RecordJsonSegmentFromBatch(g_batchState, g_batchState.result);
        DBG_LOGF("[Batch] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu", (unsigned long long)g_batchState.batchIndex, g_batchState.cached_scaled.size(), (unsigned long long)g_batchState.windowDecoded, (unsigned long long)g_batchState.windowSelected, (unsigned long long)g_batchState.scale_total_us, (unsigned long long)g_batchState.tensor_total_us, (unsigned long long)g_batchState.inference_us);
        g_batchState.reset();
        pipeline_start_us = std::chrono::steady_clock::now();
    }
    av_frame_free(&frame);
    av_frame_free(&frameBGRA);
    av_packet_free(&packet);

    auto sw_total_end_tp = std::chrono::steady_clock::now();
    g_sw_pipeline_total_us = std::chrono::duration_cast<std::chrono::microseconds>(sw_total_end_tp - sw_total_start_tp).count();
}

#ifdef _WIN32
void decode_frames(AVFormatContext *format_context,
                   AVCodecContext *codec_context,
                   int stream_index)
{
    auto hw_total_start_tp = std::chrono::steady_clock::now();
    AVPacket packet;
    AVFrame *frame = av_frame_alloc();

    int frameCount = 0;
    int infoLogCount = 0;
    auto &frameProfiler = prof::FrameProfiler::Get();
    FrameSelector selector(g_fsConfig);
    AVRational time_base = format_context->streams[stream_index]->time_base;

    auto pipeline_start_us = std::chrono::steady_clock::now();
    while (av_read_frame(format_context, &packet) >= 0)
    {
        if (packet.stream_index == stream_index)
        {
            if (avcodec_send_packet(codec_context, &packet) < 0)
            {
                std::cerr << "Error sending packet to decoder." << std::endl;
                continue;
            }
            frameProfiler.BeginFrame(frameCount);
            frameProfiler.MarkStageBegin(prof::Stage::Pipeline);
            frameProfiler.MarkStageBegin(prof::Stage::Decode);
            while (avcodec_receive_frame(codec_context, frame) == 0)
            {
                frameProfiler.MarkStageEnd(prof::Stage::Decode);
                if (g_batchConfig.new_batch_mode)
                {
                    const auto &fr = frameProfiler.Current();
                    const auto &sr = fr.stages[(int)prof::Stage::Decode];
                    if (sr.end_us >= sr.start_us)
                        g_batchState.decode_total_us += (sr.end_us - sr.start_us);
                }
                ++frameCount;
                if (g_batchConfig.new_batch_mode)
                {
                    g_batchState.windowDecoded++;
                }

                AVFrame *selectedFrame = nullptr;
                bool key = (frame->flags & AV_FRAME_FLAG_KEY) != 0;
                bool sceneCut = false;
                if (!selector.AcceptDecodedFrame(frame, time_base, selectedFrame))
                {
                    frameProfiler.SetSelectionFlags(false, key, sceneCut);
                    frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
                    frameProfiler.EndFrameAndWrite();
                    frameProfiler.MarkStageBegin(prof::Stage::Decode);
                    continue;
                }
                AVFrame *srcForVPP = selectedFrame;
                frameProfiler.SetSelectionFlags(true, key, sceneCut);
                if (g_batchConfig.new_batch_mode)
                {
                    double pts_sec = GetFramePtsSeconds(selectedFrame, time_base);
                    UpdateBatchSegmentTiming(pts_sec);
                }

                if (srcForVPP->hw_frames_ctx)
                {
                    AVHWFramesContext *hw_frames_ctx =
                        (AVHWFramesContext *)srcForVPP->hw_frames_ctx->data;
                    if (srcForVPP->format == hw_frames_ctx->format)
                    {
                        if (codec_context->hw_device_ctx)
                        {
                            AVHWDeviceContext *hw_device_ctx =
                                (AVHWDeviceContext *)codec_context->hw_device_ctx->data;
                            if (hw_device_ctx->type == AV_HWDEVICE_TYPE_D3D11VA)
                            {
                                AVHWFramesContext *hw_frames_ctx_inner =
                                    (AVHWFramesContext *)srcForVPP->hw_frames_ctx->data;
                                AVD3D11VADeviceContext *d3d11_ctx =
                                    (AVD3D11VADeviceContext *)hw_frames_ctx_inner->device_ctx->hwctx;
                                ID3D11Texture2D *dectexture = (ID3D11Texture2D *)srcForVPP->data[0];
                                int arrayindex = (int)(intptr_t)srcForVPP->data[1];

                                if (g_batchConfig.new_batch_mode)
                                {
                                    if (m_pD3D11Device == NULL)
                                    {
                                        D3D11_TEXTURE2D_DESC initDesc;
                                        dectexture->GetDesc(&initDesc);
                                        dectexture->GetDevice(&m_pD3D11Device);
                                        m_pD3D11Device->GetImmediateContext(&m_pD3D11DeviceContext);
                                        m_texturePool.Initialize(MAX_FRAME_QUEUE);
                                        D3D11_TEXTURE2D_DESC poolDesc{};
                                        poolDesc.Width = g_commonConfig.vpp_down_width;
                                        poolDesc.Height = g_commonConfig.vpp_down_height;
                                        poolDesc.MipLevels = 1;
                                        poolDesc.ArraySize = 1;
                                        poolDesc.Format = initDesc.Format;
                                        if (initDesc.Format != DXGI_FORMAT_P010 && initDesc.Format != DXGI_FORMAT_NV12)
                                        {
                                            poolDesc.Format = DXGI_FORMAT_NV12;
                                        }
                                        poolDesc.SampleDesc.Count = 1;
                                        poolDesc.Usage = D3D11_USAGE_DEFAULT;
                                        poolDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                                        poolDesc.CPUAccessFlags = 0;
                                        poolDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
                                        for (int i = 0; i < MAX_FRAME_QUEUE; i++)
                                        {
                                            ID3D11Texture2D *tex = nullptr;
                                            HRESULT hr = m_pD3D11Device->CreateTexture2D(&poolDesc, nullptr, &tex);
                                            if (hr == S_OK)
                                                m_texturePool.SetTexture(i, tex);
                                        }
                                        poolDesc.Width = initDesc.Width;
                                        poolDesc.Height = initDesc.Height;
                                        ID3D11Texture2D *tempTex = nullptr;
                                        HRESULT hr = m_pD3D11Device->CreateTexture2D(&poolDesc, nullptr, &tempTex);
                                        if (hr == S_OK)
                                            m_temp_texture = tempTex;
                                        pVPPTester = new CVPPTest();
                                        Options opts;
                                        opts.bNoSSA = false;
                                        opts.dstWidth = g_commonConfig.vpp_down_width;
                                        opts.dstHeight = g_commonConfig.vpp_down_height;
                                        switch (initDesc.Format)
                                        {
                                        case DXGI_FORMAT_NV12:
                                            opts.inFourCC = MFX_FOURCC_NV12;
                                            break;
                                        case DXGI_FORMAT_P010:
                                            opts.inFourCC = MFX_FOURCC_P010;
                                            break;
                                        default:
                                            opts.inFourCC = MFX_FOURCC_NV12;
                                            break;
                                        }
                                        opts.outFourCC = MFX_FOURCC_NV12;
                                        opts.srcWidth = initDesc.Width;
                                        opts.srcHeight = initDesc.Height;
                                        opts.surfaceComponent = MFX_SURFACE_COMPONENT_VPP_INPUT;
                                        opts.surfaceMode = SURFACE_MODE_SHARED;
                                        opts.surfaceType = MFX_SURFACE_TYPE_D3D11_TEX2D;
                                        opts.surfaceFlag = MFX_SURFACE_FLAG_IMPORT_SHARED;
                                        opts.pDevice = m_pD3D11Device;
                                        pVPPTester->Init(0, opts, nullptr);
                                    }
                                    D3D11_TEXTURE2D_DESC desc;
                                    dectexture->GetDesc(&desc);
                                    ID3D11Texture2D *outTex = m_texturePool.GetAvailableResource();
                                    if (outTex)
                                    {
                                        auto tScale0 = std::chrono::steady_clock::now();
                                        D3D11_BOX box{};
                                        box.left = 0; box.top = 0; box.front = 0; box.back = 1;
                                        box.right = desc.Width; box.bottom = desc.Height;
                                        m_pD3D11DeviceContext->CopySubresourceRegion(m_temp_texture, 0, 0, 0, 0, dectexture, arrayindex, &box);
                                        pVPPTester->ProcessingFrame(m_temp_texture, outTex);
                                        m_pD3D11DeviceContext->Flush();
                                        auto tScale1 = std::chrono::steady_clock::now();
                                        g_batchState.scale_total_us += std::chrono::duration_cast<std::chrono::microseconds>(tScale1 - tScale0).count();
                                        g_batchState.cached_textures.push_back(outTex);
                                        g_batchState.windowSelected++;
                                    }
                                    size_t curSize = g_batchState.cached_textures.size();
                                    bool trigger = (curSize > 0 && (curSize % g_batchConfig.batch_trigger) == 0);
                                    if (!trigger && g_batchConfig.max_cache > 0 && (int)curSize >= g_batchConfig.max_cache)
                                        trigger = true;
                                    if (trigger)
                                    {
                                        if (g_commonConfig.use_cb)
                                        {
                                            auto pipeline_end_us = std::chrono::steady_clock::now();
                                            g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
                                            RunBatchHW();
                                            g_batchState.reset();
                                            pipeline_start_us = std::chrono::steady_clock::now();
                                        }
                                        else
                                        {
                                            RunBatchHW();
                                            auto pipeline_end_us = std::chrono::steady_clock::now();
                                            g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
                                            prof::BatchAggregator::Get().RecordBatch(g_batchState.windowDecoded, g_batchState.windowSelected, g_batchState.decode_total_us, g_batchState.scale_total_us, g_batchState.tensor_total_us, g_batchState.inference_us, g_batchState.pipeline_us, g_batchState.prompt, g_batchState.result);
                                            RecordJsonSegmentFromBatch(g_batchState, g_batchState.result);
                                            DBG_LOGF("[Batch] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu", (unsigned long long)g_batchState.batchIndex, g_batchState.cached_textures.size(), (unsigned long long)g_batchState.windowDecoded, (unsigned long long)g_batchState.windowSelected, (unsigned long long)g_batchState.scale_total_us, (unsigned long long)g_batchState.tensor_total_us, (unsigned long long)g_batchState.inference_us);
                                            g_batchState.reset();
                                            pipeline_start_us = std::chrono::steady_clock::now();
                                        }
                                    }
                                    frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
                                    frameProfiler.EndFrameAndWrite();
                                    frameProfiler.MarkStageBegin(prof::Stage::Decode);
                                    av_frame_free(&selectedFrame);
                                    continue;
                                }
                                else
                                {
                                    D3D11_TEXTURE2D_DESC desc;
                                    dectexture->GetDesc(&desc);

                                    if (m_pD3D11Device == NULL)
                                    {
                                        DBG_LOG("m_pD3D11Device is NULL, trying to get it");
                                        dectexture->GetDevice(&m_pD3D11Device);
                                        m_pD3D11Device->GetImmediateContext(&m_pD3D11DeviceContext);
                                        DBG_LOGF("m_pD3D11Device =%p", m_pD3D11Device);

                                        m_texturePool.Initialize(MAX_FRAME_QUEUE);
                                        D3D11_TEXTURE2D_DESC textureDesc;
                                        ZeroMemory(&textureDesc, sizeof(textureDesc));
                                        textureDesc.Width = g_commonConfig.vpp_down_width;
                                        textureDesc.Height = g_commonConfig.vpp_down_height;
                                        textureDesc.MipLevels = 1;
                                        textureDesc.ArraySize = 1;
                                        textureDesc.Format = desc.Format;
                                        if (desc.Format != DXGI_FORMAT_P010 && desc.Format != DXGI_FORMAT_NV12)
                                        {
                                            DBG_LOGF("Attention: Wrong format!! %d", desc.Format);
                                            textureDesc.Format = DXGI_FORMAT_NV12;
                                        }
                                        textureDesc.SampleDesc.Count = 1;
                                        textureDesc.SampleDesc.Quality = 0;
                                        textureDesc.Usage = D3D11_USAGE_DEFAULT;
                                        textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                                        textureDesc.CPUAccessFlags = 0;
                                        textureDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

                                        ID3D11Texture2D* texture = nullptr;
                                        for (int index = 0; index < MAX_FRAME_QUEUE; index++)
                                        {
                                            HRESULT hr = m_pD3D11Device->CreateTexture2D(&textureDesc, nullptr, &texture);
                                            if (hr != S_OK)
                                            {
                                                DBG_LOG("Failed to create texture");
                                            }
                                            m_texturePool.SetTexture(index, texture);
                                        }

                                        textureDesc.Width = desc.Width;
                                        textureDesc.Height = desc.Height;

                                        HRESULT hr = m_pD3D11Device->CreateTexture2D(&textureDesc, nullptr, &m_temp_texture);
                                        if (hr != S_OK)
                                        {
                                            printf("Failed to create texture\n");
                                        }

                                        pVPPTester = new CVPPTest();
                                        Options opts;
                                        opts.bNoSSA = false;
                                        opts.dstWidth = g_commonConfig.vpp_down_width;
                                        opts.dstHeight = g_commonConfig.vpp_down_height;

                                        switch (desc.Format)
                                        {
                                        case DXGI_FORMAT_NV12:
                                            opts.inFourCC = MFX_FOURCC_NV12;
                                            break;
                                        case DXGI_FORMAT_P010:
                                            opts.inFourCC = MFX_FOURCC_P010;
                                            break;
                                        default:
                                            printf("Attention: Wrong format!! %d \n", desc.Format);
                                            opts.inFourCC = MFX_FOURCC_NV12;
                                            break;
                                        }

                                        opts.outFourCC = MFX_FOURCC_NV12;
                                        opts.srcWidth = desc.Width;
                                        opts.srcHeight = desc.Height;
                                        opts.surfaceComponent = MFX_SURFACE_COMPONENT_VPP_INPUT;
                                        opts.surfaceMode = SURFACE_MODE_SHARED;
                                        opts.surfaceType = MFX_SURFACE_TYPE_D3D11_TEX2D;
                                        opts.surfaceFlag = MFX_SURFACE_FLAG_IMPORT_SHARED;
                                        opts.pDevice = m_pD3D11Device;
                                        pVPPTester->Init(0, opts, nullptr);

                                        DBG_LOG("start to initialize video segment");
                                        DBG_LOG("video segment completed");
                                    }

                                    ID3D11Texture2D* pTempTex = m_texturePool.GetAvailableResource();
                                    frameProfiler.MarkStageBegin(prof::Stage::Copy2SharedTex);
                                    D3D11_BOX sourceRegion;
                                    sourceRegion.left = 0;
                                    sourceRegion.right = desc.Width;
                                    sourceRegion.top = 0;
                                    sourceRegion.bottom = desc.Height;
                                    sourceRegion.front = 0;
                                    sourceRegion.back = 1;
                                    m_pD3D11DeviceContext->CopySubresourceRegion(m_temp_texture, 0, 0, 0, 0, dectexture, arrayindex, &sourceRegion);
                                    m_pD3D11DeviceContext->Flush();

                                    frameProfiler.MarkStageEnd(prof::Stage::Copy2SharedTex);

                                    if (!g_batchConfig.new_batch_mode)
                                    {
                                        pVPPTester->ProcessingFrame(m_temp_texture, pTempTex);
                                        frameProfiler.MarkStageEnd(prof::Stage::Scale);
                                        SceneUnderstand(m_pD3D11Device, m_pD3D11DeviceContext, pTempTex);
                                        m_texturePool.ReturnResource(pTempTex);
                                    }
                                    else
                                    {
                                        m_texturePool.ReturnResource(pTempTex);
                                    }

                                    DBG_LOGF("[HW] processing selected Frame #%d success", frameCount);
                                }
                            }
                        }
                    }
                }
                av_frame_free(&selectedFrame);
                frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
                frameProfiler.EndFrameAndWrite();
                frameProfiler.MarkStageBegin(prof::Stage::Decode);
            }
        }
        av_packet_unref(&packet);
    }
    av_frame_free(&frame);
    if (g_batchConfig.new_batch_mode && g_batchConfig.flush_partial)
    {
        RunBatchHW();
        auto pipeline_end_us = std::chrono::steady_clock::now();
        g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
        prof::BatchAggregator::Get().RecordBatch(g_batchState.windowDecoded, g_batchState.windowSelected, g_batchState.decode_total_us, g_batchState.scale_total_us, g_batchState.tensor_total_us, g_batchState.inference_us, g_batchState.pipeline_us, g_batchState.prompt, g_batchState.result);
        if (!g_commonConfig.use_cb)
            RecordJsonSegmentFromBatch(g_batchState, g_batchState.result);
        DBG_LOGF("[Batch] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu", (unsigned long long)g_batchState.batchIndex, g_batchState.cached_scaled.size(), (unsigned long long)g_batchState.windowDecoded, (unsigned long long)g_batchState.windowSelected, (unsigned long long)g_batchState.scale_total_us, (unsigned long long)g_batchState.tensor_total_us, (unsigned long long)g_batchState.inference_us);
        g_batchState.reset();
        pipeline_start_us = std::chrono::steady_clock::now();
    }
    if (g_commonConfig.use_cb)
    {
        if (g_commonConfig.cb_multi_thread)
        {
            ProcessCBQueueAsync();
            StopCBThreadsIfEnabled();
        }
        else if (g_commonConfig.new_multithread)
        {
            StartNewCBProcThreadIfEnabled();
            CbLock();
            g_cb_pending_batches++;
            CbUnlock();
            CbNotifyAll();
            CbLock();
            while (g_cb_enqueued_batches < g_cb_pending_batches) {
                CbUnlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                CbLock();
            }
            CbUnlock();
            StopNewCBProcThreadIfEnabled();
        }
        else
        {
            ProcessCBQueueAndReport();
        }
    }
    auto hw_total_end_tp = std::chrono::steady_clock::now();
    g_hw_pipeline_total_us = std::chrono::duration_cast<std::chrono::microseconds>(hw_total_end_tp - hw_total_start_tp).count();
}
#else
// Linux stub - hardware decode not implemented
void decode_frames(AVFormatContext *format_context,
                   AVCodecContext *codec_context,
                   int stream_index)
{
    std::cerr << "[HW] Hardware decode not implemented on this platform" << std::endl;
}
#endif
