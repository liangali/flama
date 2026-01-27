#pragma once

// vlm-chat.h : Declarations for VLMPipeline chat utilities.
// This header exposes functions implemented in vlm-chat.cpp for converting
// D3D11 textures to OpenVINO tensors and running scene understanding.
//
// Usage:
//   #include "vlm-chat.h"
//   ov::genai::VLMPipeline& pipe = GetCachedVLMPipeline();
//   SceneUnderstand(device, context, texture);
//
// Environment:
//   Set VLM_MODEL_PATH to override default model path.

#ifdef _WIN32
#include <d3d11.h>
#endif
#ifdef ALTERNATE
#undef ALTERNATE
#endif

#include <openvino/openvino.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#include <vector>
#include <string>
#ifdef __cplusplus
extern "C" {
#endif
#include <libavutil/frame.h>
#ifdef __cplusplus
}
#endif
#include "debug.h"
// Convert a D3D11 texture to a CPU ov::Tensor in NHWC (1,H,W,4) with u8 elements.
// Throws std::runtime_error on failure.
ov::Tensor ConvertD3DTextureToOVTensorCPU(
    ID3D11Device* d3d_device,
    ID3D11DeviceContext* d3d_context,
    ID3D11Texture2D* d3d_texture);
//static void LogVLMModelDeviceInfo(const std::string& modelDir, const std::string& device);
// 记录模型与设备信息 (在 Pipeline 创建时调用)
static void LogVLMModelDeviceInfo(const std::string& modelDir, const std::string& device)
{
    DBG_LOG(std::string("[VLM] Model directory: ") + modelDir);
    DBG_LOG(std::string("[VLM] Target device: ") + device);
    DBG_LOG(std::string("[VLM] OpenVINO version: ") + ov::get_openvino_version().buildNumber);

    // 可用设备枚举
    try
    {
        ov::Core core;
        auto devs = core.get_available_devices();
        DBG_LOG("[VLM] ov::Core available devices:");
        for (auto& d : devs)
            DBG_LOG(std::string("  ") + d);
        if (devs.empty()) DBG_LOG("  (none)");
    }
    catch (const std::exception& e)
    {
        std::cerr << "[VLM] Failed to enumerate devices: " << e.what() << "\n";
    }

    auto envPrint = [](const char* name)
        {
            if (const char* v = std::getenv(name))
                DBG_LOG(std::string("[VLM] ") + name + "=" + v);
        };
    envPrint("OV_PLUGIN_PATH");
    envPrint("OV_EXTENSIONS_PATH");
    envPrint("OV_DISABLE_EXTENSIONS");
}
// Thread-safe cached VLMPipeline instance. First call initializes pipeline.
// 线程安全缓存
ov::genai::VLMPipeline& GetCachedVLMPipeline();

// Prepare input tensor from a D3D11 texture and perform generation.
// Returns 0 on success, non-zero on failure.
int SceneUnderstand(ID3D11Device* d3d_device,
                    ID3D11DeviceContext* d3d_context,
                    ID3D11Texture2D* d3d_texture);

bool CheckGenAIEnvironment(const std::string& modelDir,
                           const std::string& device = "CPU",
                           bool runTestInference = true);
int SceneUnderstandSW(ID3D11Device* d3d_device,
    ID3D11DeviceContext* d3d_context,
    const AVFrame* frameBGRA);

// Set output file for generated text results (appended as CSV rows)
void SetVLMResultFile(const std::string& path);
// Windows 宽路径支持（中文结果文件名）
void SetVLMResultFileW(const std::wstring& wpath);
// Set input file name for inclusion in results CSV
void SetVLMInputFile(const std::string& inputFile);