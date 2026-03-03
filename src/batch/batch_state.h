//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#pragma once

#include <vector>
#include <string>
#include <cstdint>

#ifdef _WIN32
#include <d3d11.h>
#endif

extern "C"
{
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

#include <openvino/openvino.hpp>
#include <openvino/genai/generation_config.hpp>

// Forward declarations
class TextureResourcePool;
class CVPPTest;

// ---------------- Batch State ----------------
struct BatchState
{
    std::vector<AVFrame *> cached;           // selected cloned frames awaiting batch scale/infer
    std::vector<AVFrame *> cached_scaled;    // selected cloned frames awaiting batch infer
#ifdef _WIN32
    std::vector<ID3D11Texture2D*> cached_textures; // HW path: scaled textures awaiting batch infer
#endif
    uint64_t windowDecoded = 0;              // decoded since last batch
    uint64_t windowSelected = 0;             // selected since last batch
    uint64_t batchIndex = 0;
    uint64_t decode_total_us = 0;            // accumulated per-frame decode time in window
    uint64_t scale_total_us = 0;             // accumulated per-frame scale time in window
    uint64_t tensor_total_us = 0;            // accumulated per-frame tensor time in window
    uint64_t inference_us = 0;               // inference time for this batch
    uint64_t pipeline_us = 0;                // pipeline start time from first frame
    std::string prompt;                      // current batch prompt
    std::string result;                      // current batch result
    double seg_start_sec = 0.0;              // segment start time in seconds
    double seg_end_sec = 0.0;                // segment end time in seconds
    bool seg_has_pts = false;                // whether segment timing has valid pts
    double next_seg_start_sec = 0.0;         // start time for next segment (continuity)
    bool isHW = false;                       // true if using hardware path (D3D11VA)
    struct SwsContext *swsCtx = nullptr;     // software scaling context

    void reset();
};

// CB Inference Parameters for continuous batching queue
struct CBInferenceParams {
    size_t batchIndex;
    std::string prompt;
    std::vector<ov::Tensor> tensors;
    bool use_video_input = true;
    ov::genai::GenerationConfig sampling_params;
    uint64_t windowDecoded = 0;
    uint64_t windowSelected = 0;
    uint64_t decode_total_us = 0;
    uint64_t scale_total_us = 0;
    uint64_t tensor_total_us = 0;
    uint64_t pipeline_us = 0;
    double seg_start_sec = 0.0;
    double seg_end_sec = 0.0;
    bool seg_has_pts = false;
};

// Global batch state instance
extern BatchState g_batchState;

// Global CB inference queue
extern std::vector<CBInferenceParams> g_cbInferenceQueue;

// Multi-video progress for key log messages
extern size_t g_currentVideoOrdinal;
extern size_t g_totalVideoCount;
std::string GetVideoProgressPrefix();

// Global D3D11 device pointers
#ifdef _WIN32
extern ID3D11Device *m_pD3D11Device;
extern ID3D11DeviceContext *m_pD3D11DeviceContext;
extern ID3D11Texture2D *m_temp_texture;
extern TextureResourcePool m_texturePool;
extern CVPPTest *pVPPTester;
#endif

// Helper functions
void FreeCachedFrames(std::vector<AVFrame *> &v);
#ifdef _WIN32
void ReturnCachedTextures(std::vector<ID3D11Texture2D*> &v);
#endif
void ResetHwResources();

// Frame processing helpers
AVFrame *ScaleFrameSW(AVFrame *src, int target_w, int target_h, struct SwsContext *&swsCtx);
ov::Tensor BGRAFrameToTensor(const AVFrame *frameBGRA);
#ifdef _WIN32
ID3D11Texture2D *CreateTextureFromBGRAFrame(ID3D11Device *device, ID3D11DeviceContext *context,
                                             const AVFrame *frameBGRA);
bool SaveTextureToNV12File(ID3D11Device *device, ID3D11DeviceContext *context,
                           ID3D11Texture2D *texture, int arrayindex, const char *filename);
#endif

// Batch execution functions
void RunBatchSW();
void RunBatchHW();
