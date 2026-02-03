//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "batch_state.h"
#include "utils/util.h"
#include "utils/debug.h"
#include "utils/profiling.h"
#include "device/texture_resource_pool.h"
#include "video/vpp.h"
#include "video/segment_timing.h"
#include "inference/vlm_chat.h"
#include "inference/continuous_batching_chat.h"

#include <iostream>
#include <chrono>
#include <cstring>
#include <stdexcept>

// Global instances
BatchState g_batchState;
std::vector<CBInferenceParams> g_cbInferenceQueue;

#ifdef _WIN32
ID3D11Device *m_pD3D11Device = nullptr;
ID3D11DeviceContext *m_pD3D11DeviceContext = nullptr;
ID3D11Texture2D *m_temp_texture = nullptr;
TextureResourcePool m_texturePool;
CVPPTest *pVPPTester = nullptr;
#endif

// Timing accumulators (used by decode_pipeline)
uint64_t g_sw_inference_total_us = 0;
uint64_t g_sw_pipeline_total_us = 0;
uint64_t g_hw_inference_total_us = 0;
uint64_t g_hw_pipeline_total_us = 0;

void FreeCachedFrames(std::vector<AVFrame *> &v)
{
    for (auto *f : v)
        av_frame_free(&f);
    v.clear();
}

#ifdef _WIN32
void ReturnCachedTextures(std::vector<ID3D11Texture2D*> &v)
{
    for (auto *t : v)
        if (t) m_texturePool.ReturnResource(t);
    v.clear();
}
#endif

void BatchState::reset()
{
    // Preserve segment end time for next segment's start (continuity)
    if (seg_has_pts) {
        next_seg_start_sec = seg_end_sec;
    }

    FreeCachedFrames(cached);
    FreeCachedFrames(cached_scaled);
#ifdef _WIN32
    ReturnCachedTextures(cached_textures);
#endif
    windowDecoded = 0;
    windowSelected = 0;
    decode_total_us = 0;
    scale_total_us = 0;
    tensor_total_us = 0;
    inference_us = 0;
    pipeline_us = 0;
    seg_start_sec = 0.0;
    seg_end_sec = 0.0;
    seg_has_pts = false;
    // Note: next_seg_start_sec is NOT reset - it carries over for continuity
}

void ResetHwResources()
{
    g_batchState.reset();
#ifdef _WIN32
    if (pVPPTester)
    {
        delete pVPPTester;
        pVPPTester = nullptr;
    }
    if (m_temp_texture)
    {
        m_temp_texture->Release();
        m_temp_texture = nullptr;
    }
    m_texturePool.ReleaseAll();
    if (m_pD3D11DeviceContext)
    {
        m_pD3D11DeviceContext->Release();
        m_pD3D11DeviceContext = nullptr;
    }
    if (m_pD3D11Device)
    {
        m_pD3D11Device->Release();
        m_pD3D11Device = nullptr;
    }
#endif
}

AVFrame *ScaleFrameSW(AVFrame *src, int target_w, int target_h, struct SwsContext *&swsCtx)
{
    if (!src)
        return nullptr;
    if (src->width <= 0 || src->height <= 0)
        return nullptr;
    if (!swsCtx)
    {
        swsCtx = sws_getContext(src->width, src->height, (AVPixelFormat)src->format,
                                target_w, target_h, AV_PIX_FMT_BGRA,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!swsCtx)
        {
            std::cerr << "[Batch] sws_getContext failed" << std::endl;
            return nullptr;
        }
    }
    AVFrame *out = av_frame_alloc();
    out->format = AV_PIX_FMT_BGRA;
    out->width = target_w;
    out->height = target_h;
    if (av_frame_get_buffer(out, 32) < 0)
    {
        std::cerr << "[Batch] av_frame_get_buffer failed" << std::endl;
        av_frame_free(&out);
        return nullptr;
    }
    if (sws_scale(swsCtx, src->data, src->linesize, 0, src->height, out->data, out->linesize) <= 0)
    {
        std::cerr << "[Batch] sws_scale failed" << std::endl;
        av_frame_free(&out);
        return nullptr;
    }
    return out;
}

ov::Tensor BGRAFrameToTensor(const AVFrame *frameBGRA)
{
    const int W = frameBGRA->width;
    const int H = frameBGRA->height;
    const uint8_t *src = frameBGRA->data[0];
    const int stride = frameBGRA->linesize[0];
    size_t rgbBytes = size_t(W) * size_t(H) * 3;
    unsigned char *rgb = new unsigned char[rgbBytes];
    for (int y = 0; y < H; ++y)
    {
        const uint8_t *row = src + y * stride;
        unsigned char *out = rgb + size_t(y) * size_t(W) * 3;
        for (int x = 0; x < W; ++x)
        {
            const uint8_t *px = row + x * 4;
            out[x * 3 + 0] = px[2];
            out[x * 3 + 1] = px[1];
            out[x * 3 + 2] = px[0];
        }
    }
    struct FrameAlloc
    {
        unsigned char *buf;
        size_t expect;
        void *allocate(size_t bytes, size_t) const
        {
            if (bytes != expect)
                throw std::runtime_error("FrameAlloc mismatch");
            return buf;
        }
        void deallocate(void *p, size_t bytes, size_t) noexcept { delete[] static_cast<unsigned char*>(p); }
        bool is_equal(const FrameAlloc &o) const noexcept { return this == &o; }
    } alloc{rgb, rgbBytes};
    return ov::Tensor(ov::element::u8, ov::Shape{1, (size_t)H, (size_t)W, 3}, alloc);
}

#ifdef _WIN32
ID3D11Texture2D *CreateTextureFromBGRAFrame(ID3D11Device *device, ID3D11DeviceContext *context,
                                             const AVFrame *frameBGRA)
{
    if (!device || !context || !frameBGRA)
        return nullptr;
    const int W = frameBGRA->width;
    const int H = frameBGRA->height;
    if (W <= 0 || H <= 0)
        return nullptr;

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = W;
    desc.Height = H;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    std::vector<uint8_t> linear(W * H * 4);
    const uint8_t *srcBase = frameBGRA->data[0];
    const int srcStride = frameBGRA->linesize[0];
    uint8_t *dst = linear.data();
    for (int y = 0; y < H; ++y)
    {
        std::memcpy(dst + y * W * 4, srcBase + y * srcStride, W * 4);
    }

    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = linear.data();
    init.SysMemPitch = W * 4;
    init.SysMemSlicePitch = 0;

    ID3D11Texture2D *tex = nullptr;
    HRESULT hr = device->CreateTexture2D(&desc, &init, &tex);
    if (FAILED(hr))
    {
        std::cerr << "[SW] CreateTexture2D failed for BGRA frame." << std::endl;
        return nullptr;
    }
    return tex;
}

bool SaveTextureToNV12File(ID3D11Device *device,
                           ID3D11DeviceContext *context,
                           ID3D11Texture2D *texture,
                           int arrayindex,
                           const char *filename)
{
    DBG_LOG("SaveTextureToNV12File 1");
    if (!device || !context || !texture || !filename)
    {
        DBG_LOG("SaveTextureToNV12File 2");
        return false;
    }
    DBG_LOG("SaveTextureToNV12File 3");

    D3D11_TEXTURE2D_DESC srcDesc;
    texture->GetDesc(&srcDesc);

    D3D11_TEXTURE2D_DESC stagingDesc = srcDesc;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags = 0;
    stagingDesc.ArraySize = 1;

    ID3D11Texture2D *stagingTexture = nullptr;
    HRESULT hr = device->CreateTexture2D(&stagingDesc, nullptr, &stagingTexture);
    if (FAILED(hr))
    {
        std::cerr << "Failed to create staging texture." << std::endl;
        return false;
    }

    context->CopySubresourceRegion(stagingTexture, 0, 0, 0, 0, texture, arrayindex, nullptr);

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = context->Map(stagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr))
    {
        std::cerr << "Failed to map staging texture." << std::endl;
        stagingTexture->Release();
        return false;
    }

    UINT width = srcDesc.Width;
    UINT height = srcDesc.Height;
    UINT ySize = width * height;
    UINT uvSize = ySize / 2;
    UINT totalSize = ySize + uvSize;

    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file)
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        context->Unmap(stagingTexture, 0);
        stagingTexture->Release();
        return false;
    }

    for (UINT y = 0; y < height; ++y)
    {
        file.write(reinterpret_cast<const char *>(reinterpret_cast<BYTE *>(mappedResource.pData) +
                                                  y * mappedResource.RowPitch),
                   width);
    }

    for (UINT y = 0; y < height / 2; ++y)
    {
        file.write(reinterpret_cast<const char *>(reinterpret_cast<BYTE *>(mappedResource.pData) +
                                                  (height + y) * mappedResource.RowPitch),
                   width);
    }

    file.close();
    context->Unmap(stagingTexture, 0);
    stagingTexture->Release();

    return true;
}
#endif

void RunBatchSW()
{
    auto &bs = g_batchState;
    if (bs.cached_scaled.empty())
        return;
    bs.batchIndex++;
    uint64_t tensor_total_us = 0, inference_us = 0;
    uint64_t t_batch_start = prof::FrameProfiler::Get().Current().frame_idx;
    std::vector<ov::Tensor> tensors;
    tensors.reserve(bs.cached_scaled.size());
    DBG_LOG(std::string("[VLM] build tensor for batch ") + std::to_string(bs.batchIndex));
    for (auto *scaled : bs.cached_scaled)
    {
        auto t2 = std::chrono::steady_clock::now();
        auto tensor = BGRAFrameToTensor(scaled);
        auto t3 = std::chrono::steady_clock::now();
        tensor_total_us += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        tensors.push_back(std::move(tensor));
        av_frame_free(&scaled);
    }
    bs.cached_scaled.clear();
    auto &frameProfiler = prof::FrameProfiler::Get();
    frameProfiler.MarkStageBegin(prof::Stage::Inference);
    DBG_LOG(std::string("[VLM] Inference for batch ") + std::to_string(bs.batchIndex));
    auto tinf0 = std::chrono::steady_clock::now();
    int useN = (int)tensors.size();
    std::string prompt;
    std::string out;
    auto &pipe = GetCachedVLMPipeline();
    int error_flag = 0;
    try
    {
        int maxFrames = std::max(1, g_batchConfig.max_frames_per_request);
        if (useN <= 1)
        {
            prompt = "请描述这张图片: <image>.";
            out = pipe.generate(prompt, useN == 1 ? ov::genai::image(tensors.back()) : ov::genai::image(tensors[0]));
        }
        else if (useN <= maxFrames)
        {
            prompt = g_commonConfig.prompt_video;
            out = pipe.generate(prompt, ov::genai::videos(tensors));
        }
        else
        {
            std::cerr << "Too many frames for inference. Frame count: " << useN << std::endl;
        }
        DBG_LOG(std::string("[VLM] Inference (batch ") + std::to_string(bs.batchIndex) + ") Output: " + out);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[Batch] Inference failed: " << ex.what() << std::endl;
        error_flag = 2;
    }
    auto tinf1 = std::chrono::steady_clock::now();
    inference_us = std::chrono::duration_cast<std::chrono::microseconds>(tinf1 - tinf0).count();
    frameProfiler.MarkStageEnd(prof::Stage::Inference);
    bs.tensor_total_us = tensor_total_us;
    bs.inference_us = inference_us;
    bs.prompt = prompt;
    bs.result = out;
    g_sw_inference_total_us += bs.inference_us;
}

// Forward declarations for CB processing (defined in cb_processing.cpp)
extern void ProcessCBQueueAsync();
extern void ProcessCBQueueAndReport();
extern void StartNewCBProcThreadIfEnabled();
extern void CbLock();
extern void CbUnlock();
extern void CbNotifyAll();
extern int g_cb_pending_batches;
extern int g_cb_enqueued_batches;
extern bool cb_threads_started;
extern bool cb_timer;
extern std::chrono::steady_clock::time_point g_hw_cb_start;
extern int cb_batch_size;

#ifdef _WIN32
void RunBatchHW()
{
    auto &bs = g_batchState;
    if (bs.cached_textures.empty())
        return;
    bs.batchIndex++;
    uint64_t tensor_total_us = 0, inference_us = 0;
    int error_flag = 0;
    std::vector<ov::Tensor> tensors;
    tensors.reserve(bs.cached_textures.size());
    for (auto *outTex : bs.cached_textures)
    {
        auto tTensor0 = std::chrono::steady_clock::now();
        ov::Tensor tensor = ConvertD3DTextureToOVTensorCPU(m_pD3D11Device, m_pD3D11DeviceContext, outTex);
        auto tTensor1 = std::chrono::steady_clock::now();
        tensor_total_us += std::chrono::duration_cast<std::chrono::microseconds>(tTensor1 - tTensor0).count();
        tensors.push_back(std::move(tensor));
        m_texturePool.ReturnResource(outTex);
    }
    bs.cached_textures.clear();

    std::cout << "run batch hw : use cb=" << g_commonConfig.use_cb << std::endl;
    if (g_commonConfig.use_cb)
    {
        auto &frameProfiler = prof::FrameProfiler::Get();
        frameProfiler.MarkStageBegin(prof::Stage::Inference);
        auto tInf0 = std::chrono::steady_clock::now();
        auto &pipe = GetCachedCBPipeline();
        auto &generation_info_collector = GetCachedGenerationInfoCollector();

        ov::genai::GenerationConfig sampling_params;
        sampling_params.max_new_tokens = 256;
        std::string prompt;
        std::string out;
        int useN = (int)tensors.size();
        int maxFrames = std::max(1, g_batchConfig.max_frames_per_request);
        try
        {
            if (useN <= 1)
            {
                prompt = "请描述这张图片: <image>.";
                DBG_LOG(std::string("[VLM] Inference (batch ") + std::to_string(bs.batchIndex));
            }
            else
            {
                prompt = g_commonConfig.prompt_video;
                if (useN <= maxFrames)
                {
                    CBInferenceParams params;
                    params.batchIndex = (size_t)bs.batchIndex;
                    params.prompt = prompt;
                    params.tensors = std::move(tensors);
                    params.sampling_params = sampling_params;
                    params.windowDecoded = bs.windowDecoded;
                    params.windowSelected = bs.windowSelected;
                    params.decode_total_us = bs.decode_total_us;
                    params.scale_total_us = bs.scale_total_us;
                    params.tensor_total_us = bs.tensor_total_us;
                    params.pipeline_us = bs.pipeline_us;
                    params.seg_start_sec = bs.seg_start_sec;
                    params.seg_end_sec = bs.seg_end_sec;
                    params.seg_has_pts = bs.seg_has_pts;
                    g_cbInferenceQueue.push_back(std::move(params));
                }
                else
                {
                    std::cerr << "Too many frames for inference. Frame count: " << useN << std::endl;
                }
                DBG_LOG(std::string("[VLM] Queued CB params (batch ") + std::to_string(bs.batchIndex) + ")");
            }
            std::cout << "run batch hw : cb_batch_size=" << g_batchConfig.cb_batch_size << std::endl;
            if(cb_batch_size!=0 && (bs.batchIndex % cb_batch_size == 0))
            {
                if(g_commonConfig.cb_multi_thread)
                {
                    ProcessCBQueueAsync();
                }
                else if (g_commonConfig.new_multithread) {
                    StartNewCBProcThreadIfEnabled();
                    CbLock();
                    g_cb_pending_batches++;
                    CbUnlock();
                    CbNotifyAll();
                    CbLock();
                    std::cout << "wait for g_cb_enqueued_batches >= g_cb_pending_batches" << std::endl;
                    // Wait implemented in cb_processing
                    while (g_cb_enqueued_batches < g_cb_pending_batches) {
                        CbUnlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        CbLock();
                    }
                    std::cout << "decode continue as g_cb_enqueued_batches >= g_cb_pending_batches" << std::endl;
                    CbUnlock();
                }
                else { ProcessCBQueueAndReport(); }
            }
        }
        catch (const std::exception &ex)
        {
            std::cerr << "[BatchHW] Inference failed: " << ex.what() << std::endl;
            error_flag = 2;
        }
        auto tInf1 = std::chrono::steady_clock::now();
        inference_us = std::chrono::duration_cast<std::chrono::microseconds>(tInf1 - tInf0).count();
        frameProfiler.MarkStageEnd(prof::Stage::Inference);
        bs.tensor_total_us = tensor_total_us;
        bs.inference_us = inference_us;
        bs.prompt = prompt;
    }
    else
    {
        auto &frameProfiler = prof::FrameProfiler::Get();
        frameProfiler.MarkStageBegin(prof::Stage::Inference);
        auto tInf0 = std::chrono::steady_clock::now();
        auto &pipe = GetCachedVLMPipeline();
        std::string prompt;
        std::string out;
        int useN = (int)tensors.size();
        int maxFrames = std::max(1, g_batchConfig.max_frames_per_request);
        try
        {
            ov::genai::GenerationConfig config;
            config.max_new_tokens = 256;
            if (useN <= 1)
            {
                prompt = "请描述这张图片: <image>.";
                out = pipe.generate(prompt, useN == 1 ? ov::genai::image(tensors.back()) : ov::genai::image(tensors[0]), ov::genai::generation_config(config));
                DBG_LOG(std::string("[VLM] Inference (batch ") + std::to_string(bs.batchIndex) + ") Output: " + out);
            }
            else if (useN <= maxFrames)
            {
                prompt = g_commonConfig.prompt_video;
                out = pipe.generate(prompt, ov::genai::videos(tensors), ov::genai::generation_config(config));
                DBG_LOG(std::string("[VLM] Inference (batch ") + std::to_string(bs.batchIndex) + ") Output: " + out);
            }
            else
            {
                std::cerr << "Too many frames for inference. Frame count: " << useN << std::endl;
            }
        }
        catch (const std::exception &ex)
        {
            std::cerr << "[BatchHW] Inference failed: " << ex.what() << std::endl;
            error_flag = 2;
        }
        auto tInf1 = std::chrono::steady_clock::now();
        inference_us = std::chrono::duration_cast<std::chrono::microseconds>(tInf1 - tInf0).count();
        frameProfiler.MarkStageEnd(prof::Stage::Inference);

        bs.tensor_total_us = tensor_total_us;
        bs.inference_us = inference_us;
        bs.prompt = prompt;
        bs.result = out;
        g_hw_inference_total_us += bs.inference_us;
    }
}
#endif
