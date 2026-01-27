
// VLM chat via openvino genai
// #include <openvino/genai/visual_language/pipeline.hpp>
#include "vlm_chat.h"
#include <filesystem>
#include <openvino/genai/tokenizer.hpp>
#include "../video/frame_selector.h" // for WideToUtf8/Utf8ToWide helpers

// For Microsoft::WRL::ComPtr smart COM pointer
#ifdef _WIN32
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

#include "../utils/profiling.h"
#include "../utils/debug.h"
#include <mutex>
#include <fstream>
#include "../utils/util.h"

// Result file handling (append per generation)
namespace
{
    std::mutex g_resMutex;
    std::unique_ptr<std::ofstream> g_resFile;
    bool g_headerWritten = false;
    std::string g_inputFileName;
    inline std::string SanitizeCSV(const std::string &in)
    {
        std::string out;
        out.reserve(in.size());
        for (char c : in)
        {
            if (c == '\n' || c == '\r')
                out.push_back(' ');
            else if (c == ',')
                out.push_back(' ');
            else
                out.push_back(c);
        }
        return out;
    }
    void AppendResultRow(size_t frameIdx, size_t batchSize, const std::string &prompt, const std::string &text)
    {
        std::lock_guard<std::mutex> lk(g_resMutex);
        if (!g_resFile || !g_resFile->is_open())
            return;
        if (!g_headerWritten)
        {
            (*g_resFile) << "frame,batch_size,input_file,prompt,result" << '\n';
            g_headerWritten = true;
        }
        (*g_resFile) << frameIdx << ',' << batchSize << ',' << SanitizeCSV(g_inputFileName) << ',' << SanitizeCSV(prompt) << ',' << SanitizeCSV(text) << '\n';
    }
}

void SetVLMResultFile(const std::string &path)
{
    std::lock_guard<std::mutex> lk(g_resMutex);
    g_resFile.reset();
    g_headerWritten = false;
    auto f = std::make_unique<std::ofstream>(path, std::ios::app);
    if (!f->is_open())
    {
        std::cerr << "[VLM] Failed to open result file: " << path << std::endl;
        return;
    }
    g_resFile = std::move(f);
    DBG_LOG(std::string("[VLM] Result output -> ") + path);
}

void SetVLMResultFileW(const std::wstring &wpath)
{
    std::lock_guard<std::mutex> lk(g_resMutex);
    g_resFile.reset();
    g_headerWritten = false;
    std::filesystem::path p(wpath);
    auto f = std::make_unique<std::ofstream>(p, std::ios::app | std::ios::binary);
    if (!f->is_open()) {
        std::wcerr << L"[VLM] Failed to open result file: " << wpath << std::endl;
        return;
    }
    g_resFile = std::move(f);
    // Log using UTF-8 conversion for consistency with existing debug system
    DBG_LOG(std::string("[VLM] Result output -> ") + WideToUtf8(wpath));
}

void SetVLMInputFile(const std::string &inputFile)
{
    std::lock_guard<std::mutex> lk(g_resMutex);
    g_inputFileName = inputFile;
}

ov::Tensor ConvertD3DTextureToOVTensorCPU(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    ID3D11Texture2D *texture)
{
    if (!device || !context || !texture)
        throw std::runtime_error("ConvertD3DTextureToOVTensorCPU: null D3D11 pointer");

    D3D11_TEXTURE2D_DESC td{};
    texture->GetDesc(&td);
    if (td.SampleDesc.Count != 1)
        throw std::runtime_error("ConvertD3DTextureToOVTensorCPU: multisample unsupported");

    const DXGI_FORMAT fmt = td.Format;
    if (fmt != DXGI_FORMAT_B8G8R8A8_UNORM &&
        fmt != DXGI_FORMAT_R8G8B8A8_UNORM &&
        fmt != DXGI_FORMAT_NV12)
        throw std::runtime_error("ConvertD3DTextureToOVTensorCPU: unsupported DXGI_FORMAT");

    // 建立 staging (TensorBuild timing handled via explicit FrameProfiler in caller)
    D3D11_TEXTURE2D_DESC stage = td;
    stage.Usage = D3D11_USAGE_STAGING;
    stage.BindFlags = 0;
    stage.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stage.MiscFlags = 0;

    ComPtr<ID3D11Texture2D> staging;
    HRESULT hr = device->CreateTexture2D(&stage, nullptr, &staging);
    if (FAILED(hr))
        throw std::runtime_error("ConvertD3DTextureToOVTensorCPU: CreateTexture2D failed");

    context->CopyResource(staging.Get(), texture);

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr))
        throw std::runtime_error("ConvertD3DTextureToOVTensorCPU: Map failed");

    const size_t W = td.Width;
    const size_t H = td.Height;
    const size_t planeBytes = W * H * 3;
   //  std::cout << "[VLM HW] Frame " << W << "x" << H << ", planeBytes=" << planeBytes << std::endl;
    // 自定义分配器（与 stb 例子风格不同，使用 new[] 释放）
    struct TexAllocator
    {
        unsigned char *buf;
        size_t expect;
        void *allocate(size_t bytes, size_t) const
        {
            if (bytes != expect)
                throw std::runtime_error("TexAllocator allocate size mismatch");
            return buf;
        }
        void deallocate(void *p, size_t bytes, size_t) noexcept
        {
            if (p != buf || bytes != expect)
            {
                // 尽量保持安静释放
            }
            delete[] buf;
        }
        bool is_equal(const TexAllocator &other) const noexcept { return this == &other; }
    };

    unsigned char *rgb = new unsigned char[planeBytes];

    auto finish = [&](bool ok)
    {
        context->Unmap(staging.Get(), 0);
        if (!ok)
        {
            delete[] rgb;
        }
    };

    if (fmt == DXGI_FORMAT_B8G8R8A8_UNORM)
    {
        for (size_t y = 0; y < H; ++y)
        {
            const uint8_t *row = static_cast<const uint8_t *>(mapped.pData) + y * mapped.RowPitch;
            uint8_t *out = rgb + y * W * 3;
            // 将一行拆包为 RGB
            for (size_t x = 0; x < W; ++x)
            {
                const uint8_t *px = row + x * 4; // B,G,R,A
                out[x * 3 + 0] = px[2];          // R
                out[x * 3 + 1] = px[1];          // G
                out[x * 3 + 2] = px[0];          // B
            }
        }
    }
    else if (fmt == DXGI_FORMAT_R8G8B8A8_UNORM)
    {
        for (size_t y = 0; y < H; ++y)
        {
            const uint8_t *row = static_cast<const uint8_t *>(mapped.pData) + y * mapped.RowPitch;
            uint8_t *out = rgb + y * W * 3;
            for (size_t x = 0; x < W; ++x)
            {
                const uint8_t *px = row + x * 4; // R,G,B,A
                out[x * 3 + 0] = px[0];
                out[x * 3 + 1] = px[1];
                out[x * 3 + 2] = px[2];
            }
        }
    }
    else
    { // NV12
        const uint8_t *yPlane = static_cast<const uint8_t *>(mapped.pData);
        const uint8_t *uvPlane = yPlane + mapped.RowPitch * H;
        for (size_t y = 0; y < H; ++y)
        {
            for (size_t x = 0; x < W; ++x)
            {
                const uint8_t Yv = yPlane[y * mapped.RowPitch + x];
                const size_t uvRow = y >> 1;
                const size_t uvCol = (x >> 1) << 1;
                const uint8_t Uv = uvPlane[uvRow * mapped.RowPitch + uvCol];
                const uint8_t Vv = uvPlane[uvRow * mapped.RowPitch + uvCol + 1];
                int c = int(Yv) - 16;
                int d = int(Uv) - 128;
                int e = int(Vv) - 128;
                // BT.601 integer transform
                int r = (298 * c + 409 * e + 128) >> 8;
                int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
                int b = (298 * c + 516 * d + 128) >> 8;
                r = r < 0 ? 0 : (r > 255 ? 255 : r);
                g = g < 0 ? 0 : (g > 255 ? 255 : g);
                b = b < 0 ? 0 : (b > 255 ? 255 : b);
                size_t o = (y * W + x) * 3;
                rgb[o + 0] = static_cast<unsigned char>(r);
                rgb[o + 1] = static_cast<unsigned char>(g);
                rgb[o + 2] = static_cast<unsigned char>(b);
            }
        }
    }

    // finish(true);

    // 构造 NHWC Tensor
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, H, W, 3},
        TexAllocator{rgb, planeBytes});
}
// Unified sampling & inference helper for both texture and BGRA paths.
// Maintains a single global buffer and frame counter.

struct SampleState
{
    std::vector<ov::Tensor> buffer; // accumulated tensors
    size_t frameCounter = 0;        // total frames seen
};
SampleState &GetSampleState()
{
    static SampleState st;
    return st;
}
// Configuration constants
constexpr size_t kInferInterval = 3; // every N frames trigger inference
constexpr size_t kBatchImages = 3;   // max images per batch
constexpr size_t kMaxKeep = 10;       // retain tail images

void ProcessSampleAndMaybeInfer(const ov::Tensor &image_tensor)
{
    // 打印 image_tensor 的 shape 和 size
    // std::cout << "[VLM] image_tensor shape: [";
    for (size_t i = 0; i < image_tensor.get_shape().size(); ++i)
    {
    DBG_LOGF("[VLM] shape dim %zu = %zu", i, image_tensor.get_shape()[i]);
        if (i + 1 < image_tensor.get_shape().size())
            DBG_LOG("[VLM] shape dim separator ,");
    }
    // std::cout << "] size: " << image_tensor.get_byte_size() << " bytes" << std::endl;
    auto &st = GetSampleState();
    if (st.buffer.empty())
        st.buffer.reserve(kMaxKeep);
    st.buffer.emplace_back(image_tensor);
    ++st.frameCounter;
    size_t frameIdx = st.frameCounter;
    if (frameIdx % kInferInterval != 0)
    {
        return; // skip this frame
    }
    ov::genai::VLMPipeline &pipe = GetCachedVLMPipeline();
    size_t available = st.buffer.size();
    size_t useN = std::min(kBatchImages, available);
    if (useN == 0)
        return;
    std::string prompt;
    std::string out;
    if (useN == 1) {
        prompt = "请描述这张图: <image>.";
        out = pipe.generate(prompt, ov::genai::image(st.buffer.back()));
    DBG_LOG(std::string("[VLM] Inference (frame ") + std::to_string(frameIdx) + ") Output: " + out);
        AppendResultRow(frameIdx, 1, prompt, out);
    } else {
        prompt = "请描述这个视频: ";
        try {
            out = pipe.generate(prompt, ov::genai::videos(st.buffer));
        } catch (const std::exception &ex) {
            std::cerr << "[VLM] Multi-image generate failed: " << ex.what() << "; fallback single image." << std::endl;
            out = pipe.generate("请描述这张图: <image>.", ov::genai::image(st.buffer.back()));
            useN = 1;
        }
    DBG_LOG(std::string("[VLM] Inference (frame ") + std::to_string(frameIdx) + ", batch=" + std::to_string(useN) + ") Output: " + out);
        AppendResultRow(frameIdx, useN, prompt, out);
       // prof::DumpCSV(false);
    }
    // Trim buffer if oversized
    if (st.buffer.size() > kMaxKeep)
    {
        st.buffer.erase(st.buffer.begin(), st.buffer.begin() + (st.buffer.size() - kMaxKeep));
    }
}



// 新增：采样函数
int ProcessSample(const ov::Tensor &image_tensor)
{
    // 打印 image_tensor 的 shape 和 size
    //std::cout << "[VLM] image_tensor shape: [";
    //for (size_t i = 0; i < image_tensor.get_shape().size(); ++i)
    //{
    //    std::cout << image_tensor.get_shape()[i];
    //    if (i + 1 < image_tensor.get_shape().size())
    //        std::cout << ", ";
    //}
    //std::cout << "] size: " << image_tensor.get_byte_size() << " bytes" << std::endl;
    auto &st = GetSampleState();
    // Trim buffer if oversized
    if (st.buffer.size() > kMaxKeep)
    {
        st.buffer.erase(st.buffer.begin(), st.buffer.begin() + (st.buffer.size() - kMaxKeep));
    }
    if (st.buffer.empty())
        st.buffer.reserve(kMaxKeep);
    st.buffer.emplace_back(image_tensor);
    ++st.frameCounter;
    // Accumulate per-frame timings for batch totals (decode/scale handled by caller's FrameProfiler marks)
    prof::BatchAggregator::Get().AccumulateFrame(prof::FrameProfiler::Get().Current());

    if (st.frameCounter % kInferInterval != 0)
    {
        return 0; // skip this frame
    }
    return 1;
}

// 新增：推理函数
void ProcessInfer(const std::vector<ov::Tensor> &buffer)
{
    ov::genai::VLMPipeline &pipe = GetCachedVLMPipeline();
    size_t useN = buffer.size();
    if (useN == 0)
        return;
    std::string prompt;
    std::string out;
    auto& frameProfiler = prof::FrameProfiler::Get();
    frameProfiler.MarkStageBegin(prof::Stage::Inference);
    if (useN == 1) {
        prompt = "请描述这张图: <image>.";
        out = pipe.generate(prompt, ov::genai::image(buffer.back()));
    DBG_LOG(std::string("[VLM] Inference frame ") + std::to_string(GetSampleState().frameCounter) + ") Output: " + out);

    } else {
        prompt = "请描述这个视频: ";
        try {
            out = pipe.generate(prompt, ov::genai::videos(buffer));
        } catch (const std::exception &ex) {
            std::cerr << "[VLM] Multi-image generate failed: " << ex.what() << "; fallback single image." << std::endl;
            out = pipe.generate("请描述这张图: <image>.", ov::genai::image(buffer.back()));
            useN = 1;
        }
    DBG_LOG(std::string("[VLM] Inference (frame ") + std::to_string(GetSampleState().frameCounter) + ", batch=" + std::to_string(useN) + ") Output: " + out);

    }
    frameProfiler.MarkStageEnd(prof::Stage::Inference);

    // Actual batch inference happened; write one batch CSV row with accumulated totals
    prof::BatchAggregator::Get().RecordAccumulatedBatch();

    AppendResultRow(GetSampleState().frameCounter, useN, prompt, out);
}

int SceneUnderstand(ID3D11Device *d3d_device,
                    ID3D11DeviceContext *d3d_context,
                    ID3D11Texture2D *d3d_texture)
{
    try
    {

        auto &frameProfiler = prof::FrameProfiler::Get();

        frameProfiler.MarkStageBegin(prof::Stage::TensorBuild);
        ov::Tensor image_tensor = ConvertD3DTextureToOVTensorCPU(
            d3d_device, d3d_context, d3d_texture);
        frameProfiler.MarkStageEnd(prof::Stage::TensorBuild);
        if (ProcessSample(image_tensor))
        {
            ProcessInfer(GetSampleState().buffer); 
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "SceneUnderstand exception: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}

// BGRA AVFrame 版本：用于软件解码得到的 BGRA 帧直接转换并参与采样/批量推理
int SceneUnderstandSW(ID3D11Device *d3d_device,
                      ID3D11DeviceContext *d3d_context,
                      const AVFrame *frameBGRA)
{
    try
    {
        auto &frameProfiler = prof::FrameProfiler::Get();
        if (!frameBGRA || !frameBGRA->data[0])
            return 0;
        const int W = frameBGRA->width;
        const int H = frameBGRA->height;
        if (W <= 0 || H <= 0)
            return 0;
        // 假设格式为 BGRA（调用方保证 swscale 输出）
        const uint8_t *src = frameBGRA->data[0];
        const int stride = frameBGRA->linesize[0];
        const size_t rgbBytes = size_t(W) * size_t(H) * 3;
    DBG_LOG(std::string("[VLM SW] Frame BGRA ") + std::to_string(W) + "x" + std::to_string(H) + ", stride=" + std::to_string(stride) + ", rgbBytes=" + std::to_string(rgbBytes));
        frameProfiler.MarkStageBegin(prof::Stage::TensorBuild);
        unsigned char *rgb = new unsigned char[rgbBytes];
        for (int y = 0; y < H; ++y)
        {
            const uint8_t *row = src + y * stride;
            unsigned char *out = rgb + size_t(y) * size_t(W) * 3;
            for (int x = 0; x < W; ++x)
            {
                const uint8_t *px = row + x * 4; // B,G,R,A
                out[x * 3 + 0] = px[2];          // R
                out[x * 3 + 1] = px[1];          // G
                out[x * 3 + 2] = px[0];          // B
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
            void deallocate(void *p, size_t bytes, size_t) noexcept { delete[] buf; }
            bool is_equal(const FrameAlloc &other) const noexcept { return this == &other; }
        } alloc{rgb, rgbBytes};
        ov::Tensor image_tensor(ov::element::u8, ov::Shape{1, (size_t)H, (size_t)W, 3}, alloc);
        frameProfiler.MarkStageEnd(prof::Stage::TensorBuild);
        if (ProcessSample(image_tensor))
        {
            ProcessInfer(GetSampleState().buffer);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "SceneUnderstand(BGRA) exception: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}

// 简单检查 openvino.genai (VLMPipeline) 开发环境是否就绪
// 返回 true=准备好；false=失败并打印原因
bool CheckGenAIEnvironment(const std::string &modelDir,
                           const std::string &device,
                           bool runTestInference)
{
    namespace fs = std::filesystem;
    DBG_LOG(std::string("[GenAI] OpenVINO Version: ") + ov::get_openvino_version().buildNumber);

    // 1. 模型目录检查
    if (!fs::exists(modelDir) || !fs::is_directory(modelDir))
    {
        std::cerr << "[GenAI] Model directory not found: " << modelDir << std::endl;
        return false;
    }

    // 2. 尝试列出主要文件（可选）
    size_t xmlCount = 0;
    for (auto &p : fs::directory_iterator(modelDir))
    {
        if (p.path().extension() == ".xml")
            ++xmlCount;
    }
    if (xmlCount == 0)
    {
    DBG_LOG("[GenAI] (Info) No .xml files detected, model may be GGUF or other format.");
    }

    // 3. 尝试创建 VLMPipeline
    std::unique_ptr<ov::genai::VLMPipeline> pipeline;
    try
    {
        pipeline = std::make_unique<ov::genai::VLMPipeline>(modelDir, device);
    DBG_LOG(std::string("[GenAI] VLMPipeline created on device: ") + device);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[GenAI] Failed to create VLMPipeline: " << ex.what() << std::endl;
        return false;
    }

    if (!runTestInference)
    {
    DBG_LOG("[GenAI] Skipping test inference (runTestInference=false).");
        return true;
    }

    //// 4. 构造一个最小假图像 (1x3x32x32) 以防模型需要视觉输入
    // ov::Tensor dummyImage(ov::element::u8, {1, 32, 32, 3}); // NHWC
    // std::memset(dummyImage.data(), 0, dummyImage.get_byte_size());

    //// 5. 进行最小生成测试
    // try {
    //     std::string prompt = "Describe the following image briefly: <image>.";
    //     auto result = pipeline->generate(prompt, ov::genai::image(dummyImage));
    //     std::cout << "[GenAI] Test inference output (truncated): "
    //               << (result.size() > 200 ? result.substr(0,200) + "..." : result)
    //               << std::endl;
    // } catch (const std::exception& ex) {
    //     std::cerr << "[GenAI] Test inference failed: " << ex.what() << std::endl;
    //     return false;
    // }

    DBG_LOG("[GenAI] Environment ready.");
    return true;
}

ov::genai::VLMPipeline& GetCachedVLMPipeline()
{
    static std::unique_ptr<ov::genai::VLMPipeline> g_pipeline;
    static std::once_flag g_once;
    std::call_once(g_once, []()
        {
            //std::string path = "D:\\models\\Qwen2.5-VL-3B-Instruct-int4-opt";
            //std::string device = "GPU"; // 若需 GPU 可改
            LogVLMModelDeviceInfo(g_vlmConfig.path, g_vlmConfig.device);
            g_pipeline = std::make_unique<ov::genai::VLMPipeline>(g_vlmConfig.path, g_vlmConfig.device); });
    return *g_pipeline;
}
