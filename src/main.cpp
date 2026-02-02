#include "video/video_segment.h"
#include <atlbase.h>
#include <d3d11.h>
#include <wincodec.h>
#include <windows.h>
#include <shellapi.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>

// Include C-style FFmpeg headers first
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
}
#ifdef _WIN32
#include <libavutil/hwcontext_d3d11va.h>
#endif

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "swscale.lib")
#include <dxgi1_2.h>
#include <stdio.h>
#include <new>
#include "video/vpp.h"
#include "utils/profiling.h"
#include "device/texture_resource_pool.h"
#include "inference/vlm_chat.h"
#include "utils/debug.h"
#include "video/frame_selector.h"
#include "inference/continuous_batching_chat.h"
// Config loader
#include "config/json_config.hpp"
#include "config/parse_options.h"
// #include "VideoSampling.h"
// #include "VideoSegmentation.h"
ID3D11Device *m_pD3D11Device = NULL;
ID3D11DeviceContext *m_pD3D11DeviceContext = NULL;
#define MAX_FRAME_QUEUE 128
#define ALIGN16(value) (((value + 15) >> 4) << 4) // round up to a multiple of 16

TextureResourcePool m_texturePool;

#pragma comment(lib, "d3d11.lib")
CVPPTest *pVPPTester = nullptr;
ID3D11Texture2D *m_temp_texture = nullptr;
// Global threads for CB pipeline engine and statistics reporter
static std::thread g_lmmEngineThread;
static std::thread g_statisticsReporterThread;
static bool cb_threads_started = false;
// New multithread: run ProcessCBQueueAndReport in a sub thread
static std::thread g_cbProcThread;
static bool new_cb_proc_running = false;
static std::atomic<bool> g_cb_stop{false};
#ifdef _WIN32
struct CbSync
{
    SRWLOCK lock;
    CONDITION_VARIABLE cv;
    CbSync()
    {
        InitializeSRWLock(&lock);
        InitializeConditionVariable(&cv);
    }
};
static CbSync& GetCbSync()
{
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    PVOID ctx = nullptr;
    InitOnceExecuteOnce(
        &once,
        [](PINIT_ONCE, PVOID, PVOID* context) -> BOOL
        {
            *context = new CbSync();
            return TRUE;
        },
        nullptr,
        &ctx);
    return *static_cast<CbSync*>(ctx);
}
static void CbLock() { AcquireSRWLockExclusive(&GetCbSync().lock); }
static void CbUnlock() { ReleaseSRWLockExclusive(&GetCbSync().lock); }
static void CbNotifyAll() { WakeAllConditionVariable(&GetCbSync().cv); }
template <typename Pred>
static void CbWait(Pred pred)
{
    while (!pred())
    {
        SleepConditionVariableSRW(&GetCbSync().cv, &GetCbSync().lock, INFINITE, 0);
    }
}
#else
static std::mutex g_cb_mutex;
static std::condition_variable g_cb_cv;
static void CbLock() { g_cb_mutex.lock(); }
static void CbUnlock() { g_cb_mutex.unlock(); }
static void CbNotifyAll() { g_cb_cv.notify_all(); }
template <typename Pred>
static void CbWait(Pred pred)
{
    std::unique_lock<std::mutex> lk(g_cb_mutex, std::adopt_lock);
    g_cb_cv.wait(lk, pred);
    lk.release();
}
#endif
static int g_cb_pending_batches = 0;
static int g_cb_enqueued_batches = 0;
static bool use_cb = false;
static int cb_batch_size = 0;
FSConfig g_fsConfig; // Default initialization
BatchConfig g_batchConfig; // New batch config global variable
CommonConfig g_commonConfig; // Global common config, default initialized
VLMConfig g_vlmConfig;
static int report_idx = 0;
static std::atomic<bool> finishGenerationThread{ false };

// ---------------- HW path: CB-style multithreading when use_cb=true ----------------
static std::atomic<bool> g_cb_finish_flag{false};

// Forward declaration for reporter
static void ReportCBQueueAsync();

// Thread func: drive the CB collector until finish flag
static void CBLmmEngineThreadFunc() {
    auto& collector = GetCachedGenerationInfoCollector();
    while (!g_cb_finish_flag.load()) {
        collector.run();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Thread func: wait for finish, then report summary
static void CBStatisticsThreadFunc() {
    while (!g_cb_finish_flag.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ReportCBQueueAsync();
}

// ---- Logging helpers for runtime global configs ----
static void LogFSConfig(const FSConfig& c)
{
    DBG_LOGF("[FSConfig] policy=%d frame_interval=%d window_seconds=%.3f max_per_window=%d min_frames_between=%d min_seconds_between=%.3f max_cached=%d remove_after_process=%d force_keyframe=%d enable_scene_cut=%d min_frames_between_scene_cut=%d min_seconds_between_scene_cut=%.3f enable_cache=%d",
        (int)c.policy,
        c.frame_interval,
        c.window_seconds,
        c.max_per_window,
        c.min_frames_between,
        c.min_seconds_between,
        c.max_cached,
        c.remove_after_process ? 1 : 0,
        c.force_keyframe ? 1 : 0,
        c.enable_scene_cut ? 1 : 0,
        c.min_frames_between_scene_cut,
        c.min_seconds_between_scene_cut,
        c.enable_cache ? 1 : 0);
}

static void LogBatchConfig(const BatchConfig& c)
{
    DBG_LOGF("[BatchConfig] new_batch_mode=%d batch_trigger=%d max_cache=%d decode_window=%d flush_partial=%d cb_batch_size=%d",
        c.new_batch_mode ? 1 : 0,
        c.batch_trigger,
        c.max_cache,
        c.decode_window,
        c.flush_partial ? 1 : 0,
        c.cb_batch_size);
}

static void LogCommonConfig(const CommonConfig& c)
{
    DBG_LOGF("[CommonConfig] debug=%d use_cb=%d cb_multi_thread=%d new_multithread=%d hw_decode=%d log_path=%s input_video_path=%s vpp_down=%dx%d",
        c.debug ? 1 : 0,
        c.use_cb ? 1 : 0,
        c.cb_multi_thread ? 1 : 0,
        c.new_multithread ? 1 : 0,
        c.hw_decode ? 1 : 0,
        c.log_path.c_str(),
        c.input_video_path.c_str(),
        c.vpp_down_width,
        c.vpp_down_height);
}

static void LogVLMConfig(const VLMConfig& c)
{
    DBG_LOGF("[VLMConfig] path=%s device=%s enable_continuous_batching=%d",
        c.path.c_str(),
        c.device.c_str(),
        c.enable_continuous_batching ? 1 : 0);
    DBG_LOG(std::string("[VLMConfig] scheduler=") + c.shedulerConfig.to_string());
}
// ---- Aggregated benchmarking totals ----
static uint64_t g_sw_inference_total_us = 0;   // Sum of inference_us across all SW batches
static uint64_t g_sw_pipeline_total_us = 0;    // Total time spent inside decode_frames_sw
static uint64_t g_hw_inference_total_us = 0;   // Sum of inference_us across all HW batches
static uint64_t g_hw_pipeline_total_us = 0;    // Total time spent inside decode_frames
static std::chrono::steady_clock::time_point g_hw_cb_start;    // Total time spent inside decode_frames
static std::chrono::steady_clock::time_point g_hw_cb_end;    // Total time spent inside decode_frames
static std::chrono::steady_clock::time_point g_hw_cb_multi_start;    // start time of inference thread
static std::chrono::steady_clock::time_point g_hw_cb_multi_end;    // end time of inference thread
static bool cb_timer = true;

// Prefer D3D11 hardware pixel format during decoding when offered
static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
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
static void FreeCachedFrames(std::vector<AVFrame *> &v)
{
    for (auto *f : v)
        av_frame_free(&f);
    v.clear();
}
static void ReturnCachedTextures(std::vector<ID3D11Texture2D*> &v)
{
    for (auto *t : v)
        if (t) m_texturePool.ReturnResource(t);
    v.clear();
}
// ---------------- New Batch Mode State ----------------
struct BatchState
{
    std::vector<AVFrame *> cached; // selected cloned frames awaiting batch scale/infer
    std::vector<AVFrame *> cached_scaled; // selected cloned frames awaiting batch infer
    std::vector<ID3D11Texture2D*> cached_textures; // HW path: scaled textures awaiting batch infer
    uint64_t windowDecoded = 0;    // decoded since last batch
    uint64_t windowSelected = 0;   // selected since last batch
    uint64_t batchIndex = 0;
    uint64_t decode_total_us = 0;        // accumulated per-frame decode time in window
    uint64_t scale_total_us = 0;         // accumulated per-frame decode time in window
    uint64_t tensor_total_us = 0;        // accumulated per-frame decode time in window
    uint64_t inference_us = 0;           // accumulated per-frame decode time in window
    uint64_t pipeline_us = 0;            // record pipeline start time from the first frame of this batch
    std::string prompt;                  // current batch prompt
    std::string result;                  // current batch result
    bool isHW = false;                   // true if using hardware path (D3D11VA)
    struct SwsContext *swsCtx = nullptr; // software scaling context, nullptr if not used

    void reset()
    {
        FreeCachedFrames(cached);
        FreeCachedFrames(cached_scaled);
        ReturnCachedTextures(cached_textures);
        windowDecoded = 0;
        windowSelected = 0; // reset window counters
        decode_total_us = 0;
        scale_total_us = 0;
        tensor_total_us = 0; // reset window counters
        inference_us = 0;
        pipeline_us = 0;
    }
};

static BatchState g_batchState; // SW path usage; HW path can extend similarly

// Store CB inference parameters to enqueue after decoding completes
struct CBInferenceParams {
    size_t batchIndex;
    std::string prompt;
    std::vector<ov::Tensor> tensors;
    ov::genai::GenerationConfig sampling_params;
    uint64_t windowDecoded = 0;
    uint64_t windowSelected = 0;
    uint64_t decode_total_us = 0;
    uint64_t scale_total_us = 0;
    uint64_t tensor_total_us = 0;
    uint64_t pipeline_us = 0;
};
static std::vector<CBInferenceParams> g_cbInferenceQueue;

// Scale one frame to target configured BGRA size (software path); returns newly allocated BGRA AVFrame* or nullptr.
static AVFrame *ScaleFrameSW(AVFrame *src, int target_w, int target_h, struct SwsContext *&swsCtx)
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

// Convert BGRA frame to ov::Tensor (reusing logic from SceneUnderstandSW but without inference)
static ov::Tensor BGRAFrameToTensor(const AVFrame *frameBGRA)
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
        void deallocate(void *p, size_t bytes, size_t) noexcept { delete[] buf; }
        bool is_equal(const FrameAlloc &o) const noexcept { return this == &o; }
    } alloc{rgb, rgbBytes};
    return ov::Tensor(ov::element::u8, ov::Shape{1, (size_t)H, (size_t)W, 3}, alloc);
}

// Run batch: scale all cached frames, build tensors, single inference, profiling aggregated externally
static void RunBatchSW()
{
    auto &bs = g_batchState;
    if (bs.cached_scaled.empty())
        return;
    bs.batchIndex++;
    uint64_t tensor_total_us = 0, inference_us = 0;
    uint64_t t_batch_start = prof::FrameProfiler::Get().Current().frame_idx; // placeholder index
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
    // Inference: reuse multi-image path
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
    // Record batch aggregated line via BatchAggregator using RecordBatch
    bs.tensor_total_us = tensor_total_us;
    bs.inference_us = inference_us;
    bs.prompt = prompt;
    bs.result = out;
    // Accumulate SW inference total
    g_sw_inference_total_us += bs.inference_us;
}
// New multithread helpers: start/stop ProcessCBQueueAndReport in a sub thread
static void StartNewCBProcThreadIfEnabled() {
    if (!g_commonConfig.use_cb) return;
    if (!g_commonConfig.new_multithread) return;
    if (g_commonConfig.cb_multi_thread) return; // avoid overlapping with existing CB threads
    if (new_cb_proc_running) return;
    std::cout << "[CB] Starting new-multithread (ProcessCBQueueAndReport)..." << std::endl;
    std::cout << "[CB] Mode: new_multithread (pending/enqueued handshake active)" << std::endl;
    g_cb_stop.store(false);
	g_hw_cb_multi_start = std::chrono::steady_clock::now();
    g_cbProcThread = std::thread([]() {
        auto& pipe = GetCachedCBPipeline();
        auto& generation_info_collector = GetCachedGenerationInfoCollector();
        generation_info_collector.set_start_time(std::chrono::steady_clock::now());
        while (true) {
            CbLock();
            CbWait([]() { return g_cb_stop.load() || (g_cb_pending_batches > g_cb_enqueued_batches); });
            if (g_cb_stop.load() && !(g_cb_pending_batches > g_cb_enqueued_batches)) {
                CbUnlock();
                break;
            }
            // copy and clear current queued requests under lock
            std::vector<CBInferenceParams> local_queue;
            local_queue.swap(g_cbInferenceQueue);
            CbUnlock();

            // mark CB start timer once
            if (cb_timer) {
                g_hw_cb_start = std::chrono::steady_clock::now();
                cb_timer = false;
            }

            // enqueue requests
            int req_i = 0;
            std::vector<ov::Tensor> empty_images;
            for (auto& req : local_queue) {
                DBG_LOG(std::string("[CB new-mt] Enqueue request batchIndex=") + std::to_string(req.batchIndex));
                generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, empty_images, req.tensors, req.sampling_params, false);
                ++req_i;
            }
            // signal decode thread that add_generation is complete for this batch
            CbLock();
            g_cb_enqueued_batches = g_cb_pending_batches;
            CbUnlock();
            CbNotifyAll();

            // step pipeline until all finished for current set
            while (pipe.has_non_finished_requests()) {
                pipe.step();
            }

            // report results
            g_hw_cb_end = std::chrono::steady_clock::now();
            const uint64_t cb_inference_us = std::chrono::duration_cast<std::chrono::microseconds>(g_hw_cb_end - g_hw_cb_start).count();
            g_hw_inference_total_us += cb_inference_us;
            auto tokenizer = pipe.get_tokenizer();
            size_t i = 0;
            for (auto& generation_info : generation_info_collector.generations_info) {
                auto outputs = generation_info.generation_handle->read_all();
                std::string combined;
                combined.reserve(1024);
                for (size_t k = 0; k < outputs.size(); ++k) {
                    auto text = tokenizer.decode(outputs[k].generated_ids);
                    if (!combined.empty()) combined += "\n";
                    combined += text;
                }
                DBG_LOG(std::string("[rslt_cb new-mt]") + combined);
                if (i < local_queue.size())
                {
                    const auto& req = local_queue[i];
                    prof::BatchAggregator::Get().RecordBatch(
                        req.windowDecoded,
                        req.windowSelected,
                        req.decode_total_us,
                        req.scale_total_us,
                        req.tensor_total_us,
                        cb_inference_us,
                        req.pipeline_us,
                        req.prompt,
                        combined);
                }
                ++i;
            }
            generation_info_collector.generations_info.clear();
        }
        });
    new_cb_proc_running = true;
}
static void StopNewCBProcThreadIfEnabled() {
    if (!new_cb_proc_running) return;
    g_cb_stop.store(true);
    CbNotifyAll();
    if (g_cbProcThread.joinable()) g_cbProcThread.join();
    new_cb_proc_running = false;
	g_hw_cb_multi_end = std::chrono::steady_clock::now();
    g_hw_inference_total_us = std::chrono::duration_cast<std::chrono::microseconds>(g_hw_cb_multi_end - g_hw_cb_multi_start).count();

    std::cout << "[CB] new-multithread finished. total inference time : " << g_hw_inference_total_us /1000.0<< std::endl;
}

static void StartCBThreadsIfEnabled() {
    if (!g_commonConfig.use_cb) return;
    if (!g_commonConfig.cb_multi_thread) return; // threads disabled by config
    if (cb_threads_started) return;
    std::cout << "[CB] Starting CB engine and statistics reporter threads..." << std::endl;

    // reset flags and start time
    g_cb_finish_flag.store(false);
    auto& collector = GetCachedGenerationInfoCollector();
    auto& pipe = GetCachedCBPipeline();
    collector.set_start_time(std::chrono::steady_clock::now());
    // launch threads
   // g_lmmEngineThread = std::thread(CBLmmEngineThreadFunc);
    g_lmmEngineThread = std::thread(CBPipeline::llmEngineLoop, &pipe, nullptr, &finishGenerationThread);
    g_statisticsReporterThread = std::thread(CBPipeline::statisticsReporter, &collector, 0);
    cb_threads_started = true;
}

static void StopCBThreadsIfEnabled() {
    if (!g_commonConfig.use_cb) return;
    if (!g_commonConfig.cb_multi_thread) return; // threads disabled by config
    if (!cb_threads_started) return;
    std::cout << "[CB] Stopping CB engine and statistics reporter threads..." << std::endl;
    g_cb_finish_flag.store(true);
    if (g_statisticsReporterThread.joinable()) g_statisticsReporterThread.join();
    finishGenerationThread = true;

    if (g_lmmEngineThread.joinable()) g_lmmEngineThread.join();
    cb_threads_started = false;
}


// Process queued Continuous Batching requests, step the pipeline
static void ProcessCBQueueAsync()
{
    auto cb_start = std::chrono::high_resolution_clock::now();
    if (cb_timer)
    {
        g_hw_cb_start = std::chrono::high_resolution_clock::now();
        cb_timer = false;
    }
    auto& pipe = GetCachedCBPipeline();
    auto& generation_info_collector = GetCachedGenerationInfoCollector();
    generation_info_collector.set_start_time(std::chrono::steady_clock::now());

    int req_i = 0;
    std::cout << "Processing " << g_batchConfig.cb_batch_size << " CB requests..." << std::endl;
    std::vector<ov::Tensor> empty_images;

        for (auto& req : g_cbInferenceQueue)
        {
            DBG_LOG(std::string("Enqueue CB request batchIndex=") + std::to_string(req.batchIndex));
            
            generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, empty_images, req.tensors, req.sampling_params, false);

        }

    auto cb_end = std::chrono::high_resolution_clock::now();

    std::cout << "Number of batch:" << req_i << std::endl;
    std::cout << "CB total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cb_end - cb_start).count() << " ms" << std::endl;
    StartCBThreadsIfEnabled();
    // TODO:vicky Release queued CB inference params ???
    g_cbInferenceQueue.clear();
    g_cbInferenceQueue.shrink_to_fit();
}
// Process queued Continuous Batching requests, step the pipeline, and report results
static void ReportCBQueueAsync()
{
    auto& pipe = GetCachedCBPipeline();
    auto& generation_info_collector = GetCachedGenerationInfoCollector();
 
    while (pipe.has_non_finished_requests())
    {
        pipe.step();
    }
    g_hw_cb_end = std::chrono::high_resolution_clock::now();
    g_hw_inference_total_us = std::chrono::duration_cast<std::chrono::microseconds>(g_hw_cb_end - g_hw_cb_start).count();
    ov::genai::PipelineMetrics metrics = pipe.get_metrics();
    std::cout << "CB Metrics:" << std::endl;
    std::cout << "Cache Usage: " << metrics.cache_usage << std::endl;
    std::cout << "metrics.max_cache_usage:" << metrics.max_cache_usage << std::endl;
    std::cout << "Scheduled Requests:" << metrics.scheduled_requests << std::endl;
    std::cout << "inference_duration:" << metrics.inference_duration << std::endl;
    std::cout << "Number of batch:" << generation_info_collector.num_finished << std::endl;
    std::cout << "g_hw_inference_total_us:" << g_hw_inference_total_us << std::endl;
    auto tokenizer = pipe.get_tokenizer();
    int i = 0;
    for (auto& generation_info : generation_info_collector.generations_info)
    {
        auto outputs = generation_info.generation_handle->read_all();
        std::string combined;
        combined.reserve(1024);
        for (size_t k = 0; k < outputs.size(); ++k)
        {
            auto text = tokenizer.decode(outputs[k].generated_ids);
            if (!combined.empty())
                combined += "\n";
            combined += text;
        }
        DBG_LOG(std::string("[rslt_cb of batch NO. ") + std::to_string(i + 1) + ":]\n " + combined);
        i++;
    }

}
// Process queued Continuous Batching requests, step the pipeline, and report results
static void ProcessCBQueueAndReport()
{
	// multi-thread: receive signals from video preprocess thread that all CB requests are enqueued
	// multi-thread: receive signals from self thread that should process queued requests now
    auto cb_start = std::chrono::high_resolution_clock::now();
    auto& pipe = GetCachedCBPipeline();
    auto& generation_info_collector = GetCachedGenerationInfoCollector();
    generation_info_collector.set_start_time(std::chrono::steady_clock::now());

    int req_i = 0;
	std::cout << "Processing " << g_batchConfig.cb_batch_size << " CB requests..." << std::endl;
    std::vector<ov::Tensor> empty_images;
    

        if (cb_batch_size == 1) // send them one by one
        {
            for (auto& req : g_cbInferenceQueue)
            {
                DBG_LOG(std::string("Enqueue CB request batchIndex=") + std::to_string(req.batchIndex));
                {
                    generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, empty_images, req.tensors, req.sampling_params, false);
                }
                // multi-thread: signal video preprocess thread to produce next batch
                {
                    auto handle = generation_info_collector.generations_info[req_i].generation_handle;
                    while (handle->get_status() != ov::genai::GenerationStatus::FINISHED)
                        pipe.step();
                }
                req_i++;
            }

        }
        else // if(cb_batch_size == 0)
        { // send them all at once
            for (auto& req : g_cbInferenceQueue)
            {
                DBG_LOG(std::string("Enqueue CB request batchIndex=") + std::to_string(req.batchIndex));

                // generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, req.tensors, req.sampling_params, false);
                generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, empty_images, req.tensors, req.sampling_params, false);
                req_i++;
            }
            // multi-thread: signal video preprocess thread to produce next batch
            while (pipe.has_non_finished_requests())
            {
                pipe.step();
            }
            
        }

    while (pipe.has_non_finished_requests())
    {
        pipe.step();
    }
    // multi-thread:signal inference thread to process next request
    auto cb_end = std::chrono::high_resolution_clock::now();

    ov::genai::PipelineMetrics metrics = pipe.get_metrics();
    std::cout << "CB Metrics:" << std::endl;
    std::cout << "Cache Usage: " << metrics.cache_usage << std::endl;
    std::cout << "metrics.max_cache_usage:" << metrics.max_cache_usage << std::endl;
    std::cout << "Scheduled Requests:" << metrics.scheduled_requests << std::endl;
    std::cout << "Number of batch:" << req_i << std::endl;
    std::cout << "current CB total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cb_end - cb_start).count() << " ms" << std::endl;
    g_hw_inference_total_us += std::chrono::duration_cast<std::chrono::microseconds>(cb_end - cb_start).count();
    auto tokenizer = pipe.get_tokenizer();
    const uint64_t cb_inference_us = std::chrono::duration_cast<std::chrono::microseconds>(cb_end - cb_start).count();
    size_t i = 0;
    for (auto& generation_info : generation_info_collector.generations_info)
    {
        report_idx++;
        auto outputs = generation_info.generation_handle->read_all();
        std::string combined;
        combined.reserve(1024);
        for (size_t k = 0; k < outputs.size(); ++k)
        {
            auto text = tokenizer.decode(outputs[k].generated_ids);
            if (!combined.empty())
                combined += "\n";
            combined += text;
        }
        DBG_LOG(std::string("[rslt_cb of batch NO. ") + std::to_string(report_idx) + ":]\n " + combined);
        if (i < g_cbInferenceQueue.size())
        {
            const auto& req = g_cbInferenceQueue[i];
            prof::BatchAggregator::Get().RecordBatch(
                req.windowDecoded,
                req.windowSelected,
                req.decode_total_us,
                req.scale_total_us,
                req.tensor_total_us,
                cb_inference_us,
                req.pipeline_us,
                req.prompt,
                combined);
        }
        ++i;
        
    }
    // Release queued CB inference params now that processing is complete
    generation_info_collector.generations_info.clear();
    g_cbInferenceQueue.clear();
    g_cbInferenceQueue.shrink_to_fit();
}

// Hardware batch execution for D3D11 decoded frames
static void RunBatchHW(/*bool use_cb*/)
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

    // start inference
    std::cout << "run batch hw : use cb=" << g_commonConfig.use_cb << std::endl;
    if (g_commonConfig.use_cb) // continuous batching mode
    {
        auto &frameProfiler = prof::FrameProfiler::Get();
        frameProfiler.MarkStageBegin(prof::Stage::Inference);
        auto tInf0 = std::chrono::steady_clock::now();
        auto &pipe = GetCachedCBPipeline();
        auto &generation_info_collector = GetCachedGenerationInfoCollector();
        // Start CB engine/statistics threads on first real inference

        ov::genai::GenerationConfig sampling_params;
        sampling_params.max_new_tokens = 256;
        std::string prompt;
        std::string out;
        int useN = (int)tensors.size();
        int maxFrames = std::max(1, g_batchConfig.max_frames_per_request);
        // bs.batchIndex is global batch index; need to map to CB batch index
        try
        {
            if (useN <= 1)
            {
                prompt = "请描述这张图片: <image>.";
                // out = pipe.generate(prompt, useN == 1 ? ov::genai::image(tensors.back()) : ov::genai::image(tensors[0]));
                DBG_LOG(std::string("[VLM] Inference (batch ") + std::to_string(bs.batchIndex));
            }
            else
            {
                prompt = g_commonConfig.prompt_video;
                // For CB path, collect params and enqueue after decode completes, chunking frames
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
                    g_cbInferenceQueue.push_back(std::move(params));
                }
                else
                {
                    std::cerr << "Too many frames for inference. Frame count: " << useN << std::endl;
                }
                DBG_LOG(std::string("[VLM] Queued CB params (batch ") + std::to_string(bs.batchIndex) + ")");
                if (!cb_threads_started)
                {
                   // cb_threads_started = true;
                  //  lmmEngineThread = std::thread(CBPipeline::llmEngineLoop, &pipe, nullptr, &finishGenerationThread);
                    // statisticsReporterThread started after we add first request below
                }
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
                    // signal worker there's a pending batch and wait until enqueued
                    CbLock();
                    g_cb_pending_batches++;
                    CbUnlock();
                    CbNotifyAll();
                    CbLock();
                    std::cout << "wait for g_cb_enqueued_batches >= g_cb_pending_batches" << std::endl;
                    CbWait([](){ return g_cb_enqueued_batches >= g_cb_pending_batches; });
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
        // bs.result = out;

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
        // Accumulate HW inference total in non-CB mode
        g_hw_inference_total_us += bs.inference_us;
    }
    // prof::BatchAggregator::Get().RecordBatch(bs.windowDecoded, bs.windowSelected, bs.decode_total_us, scale_total_us, tensor_total_us, inference_us, prompt, out);
    //// AppendResultRow(bs.batchIndex, useN, prompt, out);
    // DBG_LOGF("[BatchHW] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu", (unsigned long long)bs.batchIndex, bs.cached.size(), (unsigned long long)bs.windowDecoded, (unsigned long long)bs.windowSelected, (unsigned long long)scale_total_us, (unsigned long long)tensor_total_us, (unsigned long long)inference_us);
    // FreeCachedFrames(bs.cached);
    // bs.windowDecoded = 0;
    // bs.windowSelected = 0;
    // bs.decode_total_us = 0;
}

// 函数用于将 ID3D11Texture2D 的数据以 NV12 格式保存到文件
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

    // 获取源纹理的描述信息
    D3D11_TEXTURE2D_DESC srcDesc;
    texture->GetDesc(&srcDesc);

    // 创建 staging 纹理描述
    D3D11_TEXTURE2D_DESC stagingDesc = srcDesc;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags = 0;
    stagingDesc.ArraySize = 1;

    // 创建 staging 纹理
    ID3D11Texture2D *stagingTexture = nullptr;
    HRESULT hr = device->CreateTexture2D(&stagingDesc, nullptr, &stagingTexture);
    if (FAILED(hr))
    {
        std::cerr << "Failed to create staging texture." << std::endl;
        return false;
    }

    // 将源纹理的数据复制到 staging 纹理
    // context->CopyResource(stagingTexture, texture);
    context->CopySubresourceRegion(stagingTexture, 0, 0, 0, 0, texture, arrayindex, nullptr);

    // 映射 staging 纹理以读取数据
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    hr = context->Map(stagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr))
    {
        std::cerr << "Failed to map staging texture." << std::endl;
        stagingTexture->Release();
        return false;
    }

    // 计算 NV12 数据的大小
    UINT width = srcDesc.Width;
    UINT height = srcDesc.Height;
    UINT ySize = width * height;
    UINT uvSize = ySize / 2;
    UINT totalSize = ySize + uvSize;

    // 打开文件以写入数据
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file)
    {
        std::cerr << "Failed to open file for writing." << std::endl;
        context->Unmap(stagingTexture, 0);
        stagingTexture->Release();
        return false;
    }

    // 写入 Y 平面数据
    for (UINT y = 0; y < height; ++y)
    {
        file.write(reinterpret_cast<const char *>(reinterpret_cast<BYTE *>(mappedResource.pData) +
                                                  y * mappedResource.RowPitch),
                   width);
    }

    // 写入 UV 平面数据
    for (UINT y = 0; y < height / 2; ++y)
    {
        file.write(reinterpret_cast<const char *>(reinterpret_cast<BYTE *>(mappedResource.pData) +
                                                  (height + y) * mappedResource.RowPitch),
                   width);
    }

    // 关闭文件
    file.close();

    // 解除映射
    context->Unmap(stagingTexture, 0);

    // 释放 staging 纹理
    stagingTexture->Release();

    return true;
}

// 初始化FFmpeg
void init_ffmpeg()
{
    avformat_network_init();
}

// 查找视频流
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

// 打开解码器
AVCodecContext *open_decoder(AVFormatContext *format_context,
                             int stream_index,
                             const char *hw_device_type)
{
    AVCodecParameters *codec_params = format_context->streams[stream_index]->codecpar;
    // const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264); // AV_CODEC_ID_H264
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

    // Ensure decoder prefers D3D11 hardware frames when possible
   // codec_context->get_format = get_hw_format;

    // Enlarge the HW surface pool: cached frames + headroom
    //int headroom = 16;
    //int cached = std::max(0, g_batchConfig.max_cache);
    //codec_context->extra_hw_frames = std::max(32, cached + headroom);

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

// 软件解码器：不使用硬件设备，直接打开指定流的编解码器（按原始 codecpar->codec_id）
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
    // 可设置线程数（软解可用多线程）
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

// 将 CPU 内存中的 BGRA 帧数据上传到 D3D11 纹理 (DXGI_FORMAT_B8G8R8A8_UNORM)
// 返回可供后续处理的 ID3D11Texture2D* (调用方负责 Release)
static ID3D11Texture2D *CreateTextureFromBGRAFrame(ID3D11Device *device, ID3D11DeviceContext *context,
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
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE; // 若后续需要 render target 可再加
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    // 仅一行数据初始化，需处理 stride 行间距复制
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
    return tex; // 上传完成，返回纹理
}

// 软件解码 + 转换 BGRA + 上传到 D3D11 纹理后调用 SceneUnderstand (支持 HEVC / 10-bit)
void decode_frames_sw(AVFormatContext *format_context, AVCodecContext *codec_context, int stream_index)
{
    auto sw_total_start_tp = std::chrono::steady_clock::now(); // measure entire function
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();     // 原始解码输出
    AVFrame *frameBGRA = av_frame_alloc(); // 转换后的 BGRA
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

    // Create frame selector (using global config)
    FrameSelector selector(g_fsConfig);
    AVRational time_base = format_context->streams[stream_index]->time_base;

    // Batch CSV path setup: reuse profiler naming if available
    // Assuming caller previously set FrameProfiler output; derive a sibling "-batch.csv"
    {
        // This block can be replaced with explicit path wiring if you already compute output names
        // For now, we use the same wide path if available via SetOutputFileW call in the main flow
        // Here we just ensure BatchAggregator has a file set to avoid missing outputs
        // If you already pass a specific profiling path, please set batch file there too.
        // No-op if file is already opened elsewhere.
        // Note: We cannot read the current path from FrameProfiler, so leave to main wiring.
    }
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
        // decode_start_us = frameProfiler.now_us();
        // decode_start_us_check = av_gettime();

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

        // Drain frames produced by this packet
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
            // End decode stage and accumulate decode time in batch mode
            frameProfiler.MarkStageEnd(prof::Stage::Decode);
            if (g_batchConfig.new_batch_mode)
            {
                const auto &fr = frameProfiler.Current();
                const auto &sr = fr.stages[(int)prof::Stage::Decode];
                if (sr.end_us >= sr.start_us)
                    g_batchState.decode_total_us += (sr.end_us - sr.start_us);
                g_batchState.windowDecoded++;
            }

            // 抽帧：不选中的帧直接结束 Pipeline；新批模式下选中帧进入缓存
            AVFrame *selectedFrame = nullptr;
            bool key = (frame->flags & AV_FRAME_FLAG_KEY) != 0;
            bool sceneCut = false; // TODO: integrate real scene cut detection
            if (!selector.AcceptDecodedFrame(frame, time_base, selectedFrame))
            {
                frameProfiler.SetSelectionFlags(false, key, sceneCut);
                prof::Summary::Get().RecordFrame(false, key, sceneCut);
                frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
                frameProfiler.EndFrameAndWrite();
                frameProfiler.MarkStageBegin(prof::Stage::Decode);
                continue;
            }
            AVFrame *srcForScale = selectedFrame; // legacy path variable
            frameProfiler.SetSelectionFlags(true, key, sceneCut);
            prof::Summary::Get().RecordFrame(true, key, sceneCut);

            // 首帧或变化日志
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
                // 重建 sws
                if (swsCtx)
                {
                    sws_freeContext(swsCtx);
                    swsCtx = nullptr;
                }
                // 分配/重新分配 BGRA 目标帧缓冲
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
                // Immediately scale selected frame and cache the scaled frame to reduce memory usage
                auto tScale0 = std::chrono::steady_clock::now();
                AVFrame *scaled = ScaleFrameSW(srcForScale, target_width, target_height, swsCtx);
                auto tScale1 = std::chrono::steady_clock::now();
                if (scaled)
                {
                    g_batchState.cached_scaled.push_back(scaled);
                    g_batchState.scale_total_us += std::chrono::duration_cast<std::chrono::microseconds>(tScale1 - tScale0).count();
                }
                // We no longer cache raw selected frames
                av_frame_free(&selectedFrame);
                g_batchState.windowSelected++;
                // Trigger batch if multiple of K or forced by max_cache (use scaled cache size)
                if ((int)g_batchState.cached_scaled.size() >= g_batchConfig.max_cache || ((int)g_batchState.cached_scaled.size() % g_batchConfig.batch_trigger) == 0)
                {
                    RunBatchSW();
                    auto pipeline_end_us = std::chrono::steady_clock::now();
                    g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
                    prof::BatchAggregator::Get().RecordBatch(g_batchState.windowDecoded, g_batchState.windowSelected, g_batchState.decode_total_us, g_batchState.scale_total_us, g_batchState.tensor_total_us, g_batchState.inference_us, g_batchState.pipeline_us, g_batchState.prompt, g_batchState.result);
                    DBG_LOGF("[Batch] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu pipeline_us=%llu", (unsigned long long)g_batchState.batchIndex, g_batchState.cached_scaled.size(), (unsigned long long)g_batchState.windowDecoded, (unsigned long long)g_batchState.windowSelected, (unsigned long long)g_batchState.scale_total_us, (unsigned long long)g_batchState.tensor_total_us, (unsigned long long)g_batchState.inference_us, (unsigned long long)g_batchState.pipeline_us);
                    g_batchState.reset();
                    pipeline_start_us = std::chrono::steady_clock::now();
                }
            }
            else
            {
                // Legacy immediate scale+inference path (unchanged behavior)
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
                SceneUnderstandSW(m_pD3D11Device, m_pD3D11DeviceContext, frameBGRA);
                av_frame_free(&selectedFrame);
            }

            frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
            frameProfiler.EndFrameAndWrite();
            frameProfiler.MarkStageBegin(prof::Stage::Decode); // 为下一个 receive 做准备
        }
    }

    // Flush 解码器，获取尾帧（也应用抽帧策略）
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
        // Flush 阶段的抽帧选择
        AVFrame *selectedFrame = nullptr;
        bool key = (frame->flags & AV_FRAME_FLAG_KEY) != 0;
        bool sceneCut = false; // flush 不额外触发 scene cut
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
        AVFrame *flushScaleSrc = selectedFrame;
        if (g_batchConfig.new_batch_mode)
        {
            // Scale immediately in flush
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
    // Flush remaining batch if new mode and partial flush enabled
    if (g_batchConfig.new_batch_mode && !g_batchState.cached_scaled.empty() && g_batchConfig.flush_partial)
    {
        RunBatchSW();
        auto pipeline_end_us = std::chrono::steady_clock::now();
        g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
        prof::BatchAggregator::Get().RecordBatch(g_batchState.windowDecoded, g_batchState.windowSelected, g_batchState.decode_total_us, g_batchState.scale_total_us, g_batchState.tensor_total_us, g_batchState.inference_us, g_batchState.pipeline_us, g_batchState.prompt, g_batchState.result);
        DBG_LOGF("[Batch] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu", (unsigned long long)g_batchState.batchIndex, g_batchState.cached_scaled.size(), (unsigned long long)g_batchState.windowDecoded, (unsigned long long)g_batchState.windowSelected, (unsigned long long)g_batchState.scale_total_us, (unsigned long long)g_batchState.tensor_total_us, (unsigned long long)g_batchState.inference_us);
        g_batchState.reset();
        pipeline_start_us = std::chrono::steady_clock::now();
    }
    av_frame_free(&frame);
    av_frame_free(&frameBGRA);
    av_packet_free(&packet);

    // end-to-end decode_frames_sw time
    auto sw_total_end_tp = std::chrono::steady_clock::now();
    g_sw_pipeline_total_us = std::chrono::duration_cast<std::chrono::microseconds>(sw_total_end_tp - sw_total_start_tp).count();
}



// 解码帧并获取缓冲区句柄
void decode_frames(AVFormatContext *format_context,
                   AVCodecContext *codec_context,
                   int stream_index)
{
    auto hw_total_start_tp = std::chrono::steady_clock::now(); // measure entire function
    AVPacket packet;
    AVFrame *frame = av_frame_alloc();
    // AVFrame *frame_hw = av_frame_alloc();

    int frameCount = 0;
    int infoLogCount = 0;
    // FILE *fp;
    // int err = fopen_s(&fp, "D:\\test\\VLM_test_video_clips\\000-S-三生不渡相思苦-E01-E03.mp4", "w+b");
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
                // Log first few decoded frames' pix_fmt and hw_frames_ctx
                /*if (infoLogCount < 10)
                {
                    const char *fmt_name = av_get_pix_fmt_name((AVPixelFormat)frame->format);
                    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get((AVPixelFormat)frame->format);
                    int bitDepth = desc ? desc->comp[0].depth : 0;
                    DBG_LOGF("[HW] Decoded frame #%d: fmt=%s bitdepth=%d size=%dx%d hw_ctx=%s",
                             frameCount,
                             fmt_name ? fmt_name : "<unknown>",
                             bitDepth,
                             frame->width,
                             frame->height,
                             (frame->hw_frames_ctx ? "yes" : "no"));
                }*/
                frameProfiler.MarkStageEnd(prof::Stage::Decode); // 硬解码阶段结束
                if (g_batchConfig.new_batch_mode)
                {
                    const auto &fr = frameProfiler.Current();
                    const auto &sr = fr.stages[(int)prof::Stage::Decode];
                    if (sr.end_us >= sr.start_us)
                        g_batchState.decode_total_us += (sr.end_us - sr.start_us);
                }
                ++frameCount;
               // std::cout << "decode frame#" <<frameCount << std::endl;
                if (g_batchConfig.new_batch_mode)
                {
                    g_batchState.windowDecoded++;
                }

                // 抽帧选择：未选中直接结束 Pipeline
                AVFrame *selectedFrame = nullptr;
                bool key = (frame->flags & AV_FRAME_FLAG_KEY) != 0;
                bool sceneCut = false; // TODO: integrate real scene cut detection
                if (!selector.AcceptDecodedFrame(frame, time_base, selectedFrame))
                {
                    /*if (infoLogCount < 10)
                    {
                        DBG_LOGF("[HW] Frame #%d not selected by selector", frameCount);
                        infoLogCount++;
                    }*/
                    frameProfiler.SetSelectionFlags(false, key, sceneCut);
                    frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
                    frameProfiler.EndFrameAndWrite();
                    frameProfiler.MarkStageBegin(prof::Stage::Decode);
                    continue;
                }
                AVFrame *srcForVPP = selectedFrame;
                frameProfiler.SetSelectionFlags(true, key, sceneCut);

               /* if (infoLogCount < 10)
                {
                    const char *sel_fmt = av_get_pix_fmt_name((AVPixelFormat)srcForVPP->format);
                    const AVPixFmtDescriptor *sel_desc = av_pix_fmt_desc_get((AVPixelFormat)srcForVPP->format);
                    int sel_bitDepth = sel_desc ? sel_desc->comp[0].depth : 0;
                    DBG_LOGF("[HW] Selected frame #%d: fmt=%s bitdepth=%d size=%dx%d hw_ctx=%s",
                             frameCount,
                             sel_fmt ? sel_fmt : "<unknown>",
                             sel_bitDepth,
                             srcForVPP->width,
                             srcForVPP->height,
                             (srcForVPP->hw_frames_ctx ? "yes" : "no"));
                    infoLogCount++;
                }*/

                if (srcForVPP->hw_frames_ctx)
                {
                    AVHWFramesContext *hw_frames_ctx =
                        (AVHWFramesContext *)srcForVPP->hw_frames_ctx->data;
                    if (srcForVPP->format == hw_frames_ctx->format)
                    {
                        // 获取缓冲区句柄
                        if (codec_context->hw_device_ctx)
                        {
                            AVHWDeviceContext *hw_device_ctx =
                                (AVHWDeviceContext *)codec_context->hw_device_ctx->data;
                            if (hw_device_ctx->type == AV_HWDEVICE_TYPE_D3D11VA)
                            {
                                AVHWFramesContext *hw_frames_ctx =
                                    (AVHWFramesContext *)srcForVPP->hw_frames_ctx->data;
                                AVD3D11VADeviceContext *d3d11_ctx =
                                    (AVD3D11VADeviceContext *)hw_frames_ctx->device_ctx->hwctx;
                                ID3D11Texture2D *dectexture = (ID3D11Texture2D *)srcForVPP->data[0];
                                int arrayindex = (int)srcForVPP->data[1];
                                // New batch mode path: cache selected HW frames & trigger batch processing
                                if (g_batchConfig.new_batch_mode)
                                {
                                    // Lazy initialize hardware resources (simplified)
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
                                    // Scale immediately and cache the scaled texture to reduce memory usage
                                    D3D11_TEXTURE2D_DESC desc;
                                    dectexture->GetDesc(&desc);
                                    ID3D11Texture2D *outTex = m_texturePool.GetAvailableResource();
                                    if (outTex)
                                    {
                                        auto tScale0 = std::chrono::steady_clock::now();
                                        // Copy source into shared temp texture
                                        D3D11_BOX box{};
                                        box.left = 0; box.top = 0; box.front = 0; box.back = 1;
                                        box.right = desc.Width; box.bottom = desc.Height;
                                        m_pD3D11DeviceContext->CopySubresourceRegion(m_temp_texture, 0, 0, 0, 0, dectexture, arrayindex, &box);
                                        // Run VPP to scale into outTex
										// std::cout << "do vpp" << std::endl;
                                        pVPPTester->ProcessingFrame(m_temp_texture, outTex);
                                        m_pD3D11DeviceContext->Flush();
                                        auto tScale1 = std::chrono::steady_clock::now();
                                        g_batchState.scale_total_us += std::chrono::duration_cast<std::chrono::microseconds>(tScale1 - tScale0).count();
                                        g_batchState.cached_textures.push_back(outTex); // retain until batch
                                        g_batchState.windowSelected++;
                                    }
                                    // Trigger batch using cached scaled textures size
                                    size_t curSize = g_batchState.cached_textures.size();
                                    bool trigger = (curSize > 0 && (curSize % g_batchConfig.batch_trigger) == 0);
                                    if (!trigger && g_batchConfig.max_cache > 0 && curSize >= g_batchConfig.max_cache)
                                        trigger = true;
                                    if (trigger)
                                    {
                                        if (g_commonConfig.use_cb)
                                        {
                                            auto pipeline_end_us = std::chrono::steady_clock::now();
                                            g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
                                            RunBatchHW(/*use_cb*/);
                                            g_batchState.reset();
                                            pipeline_start_us = std::chrono::steady_clock::now();
                                        }
                                        else
                                        {
                                            RunBatchHW(/*use_cb*/);
                                            auto pipeline_end_us = std::chrono::steady_clock::now();
                                            g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
                                            prof::BatchAggregator::Get().RecordBatch(g_batchState.windowDecoded, g_batchState.windowSelected, g_batchState.decode_total_us, g_batchState.scale_total_us, g_batchState.tensor_total_us, g_batchState.inference_us, g_batchState.pipeline_us, g_batchState.prompt, g_batchState.result);
                                            DBG_LOGF("[Batch] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu", (unsigned long long)g_batchState.batchIndex, g_batchState.cached_textures.size(), (unsigned long long)g_batchState.windowDecoded, (unsigned long long)g_batchState.windowSelected, (unsigned long long)g_batchState.scale_total_us, (unsigned long long)g_batchState.tensor_total_us, (unsigned long long)g_batchState.inference_us);
                                            g_batchState.reset();
                                            pipeline_start_us = std::chrono::steady_clock::now();
                                        }
                                    }
                                    // End pipeline early (no per-frame inference timings)
                                    frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
                                    frameProfiler.EndFrameAndWrite();
                                    frameProfiler.MarkStageBegin(prof::Stage::Decode);
                                    av_frame_free(&selectedFrame);
                                    continue; // proceed to next received frame
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

                                        // m_pvideoSampler = new VideoSampling(m_pD3D11Device, video_sampling_output, 30, 30, textureDesc.Width, textureDesc.Height);
                                        // m_pvideoSegmenter = new VideoSegmentation(m_pD3D11Device, video_sampling_output, video_segment_output);

                                        m_texturePool.Initialize(MAX_FRAME_QUEUE);
                                        D3D11_TEXTURE2D_DESC textureDesc;
                                        ZeroMemory(&textureDesc, sizeof(textureDesc));
                                        textureDesc.Width = g_commonConfig.vpp_down_width;
                                        // desc.Width;
                                        textureDesc.Height = g_commonConfig.vpp_down_height;
                                        // desc.Height;
                                        textureDesc.MipLevels = 1;
                                        textureDesc.ArraySize = 1;
                                        textureDesc.Format = desc.Format;
                                        if (desc.Format != DXGI_FORMAT_P010 && desc.Format != DXGI_FORMAT_NV12)
                                        {
                                            DBG_LOGF("Attention: Wrong format!! %d", desc.Format);
                                            textureDesc.Format = DXGI_FORMAT_NV12; // hevc
                                        }
                                        // textureDesc.Format = DXGI_FORMAT_NV12; // DXGI_FORMAT_B8G8R8A8_UNORM;
                                        textureDesc.SampleDesc.Count = 1;
                                        textureDesc.SampleDesc.Quality = 0;
                                        textureDesc.Usage = D3D11_USAGE_DEFAULT;
                                        textureDesc.BindFlags =
                                            D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                                        textureDesc.CPUAccessFlags = 0;
                                        textureDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

                                        ID3D11Texture2D* texture = nullptr;
                                        // Create Sampler output buffer pool
                                        for (int index = 0; index < MAX_FRAME_QUEUE; index++)
                                        {

                                            HRESULT hr = m_pD3D11Device->CreateTexture2D(&textureDesc,
                                                nullptr,
                                                &texture);
                                            if (hr != S_OK)
                                            {
                                                DBG_LOG("Failed to create texture");
                                            }
                                            m_texturePool.SetTexture(index, texture);
                                        }

                                        textureDesc.Width = desc.Width;
                                        textureDesc.Height = desc.Height;

                                        HRESULT hr = m_pD3D11Device->CreateTexture2D(&textureDesc,
                                            nullptr,
                                            &m_temp_texture);
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
                                            opts.inFourCC = MFX_FOURCC_NV12; // h264
                                            break;
                                        case DXGI_FORMAT_P010:
                                            opts.inFourCC = MFX_FOURCC_P010; // hevc
                                            break;
                                        default:
                                            printf("Attention: Wrong format!! %d \n", desc.Format);
                                            opts.inFourCC = MFX_FOURCC_NV12; // hevc
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
                                        // opts.outfileName = "D:\\temp\\dump.argb";
                                        pVPPTester->Init(0, opts, nullptr);

                                        // VideoSegParams params;
                                        // params.bZeroCopy = TRUE;
                                        // params.pD3D11Device = m_pD3D11Device;
                                        // params.inmodelName = (char *)("D:\\models\\resnet-50-ov-int4\\");

                                        DBG_LOG("start to initialize video segment");
                                        // InitVideoSegment(params);

                                        DBG_LOG("video segment completed");
                                    }

                                    // printf("start to do processing\n");

                                    ID3D11Texture2D* pTempTex = m_texturePool.GetAvailableResource();
                                    frameProfiler.MarkStageBegin(prof::Stage::Copy2SharedTex);
                                    D3D11_BOX sourceRegion;
                                    sourceRegion.left = 0;
                                    sourceRegion.right = desc.Width;
                                    sourceRegion.top = 0;
                                    sourceRegion.bottom = desc.Height;
                                    sourceRegion.front = 0;
                                    sourceRegion.back = 1;
                                    m_pD3D11DeviceContext->CopySubresourceRegion(m_temp_texture,
                                        0,
                                        0,
                                        0,
                                        0,
                                        dectexture,
                                        arrayindex,
                                        &sourceRegion);
                                    m_pD3D11DeviceContext->Flush();

                                    frameProfiler.MarkStageEnd(prof::Stage::Copy2SharedTex);
                                    // TODO: if place pTempTex acquisition here, the Scale, Inference,pipeline time will be super long, why?
                                    // ID3D11Texture2D* pTempTex = m_texturePool.GetAvailableResource(); // vpp  output

                                    if (!g_batchConfig.new_batch_mode)
                                    {
                                        // Legacy single-frame path
                                        pVPPTester->ProcessingFrame(m_temp_texture, pTempTex);
                                        frameProfiler.MarkStageEnd(prof::Stage::Scale);
                                        SceneUnderstand(m_pD3D11Device, m_pD3D11DeviceContext, pTempTex);
                                        m_texturePool.ReturnResource(pTempTex);
                                    }
                                    else
                                    {
                                        // Should not execute in new batch mode; return texture
                                        m_texturePool.ReturnResource(pTempTex);
                                    }

                                    DBG_LOGF("[HW] processing selected Frame #%d success", frameCount);

                                    /*SaveTextureToNV12File(
                                        m_pD3D11Device,
                                        m_pD3D11DeviceContext,
                                        pTempTex,
                                        0,
                                        "D:\\temp\\dump_1234.argb");*/

                                        // m_pMfxScreenEnc->EncodeTexture(pTempTex, pBitstream, nMaxLen, &len);
                                        // if (fp) {
                                        //     fwrite(pBitstream, 1, len, fp);
                                        //     fflush(fp);
                                        // }
                                        //
                                        // m_pvideoSampler->sampler((GenericTexture)dectexture);
                                        // video_phase1_sampling(dectexture);
                                        // std::cout << "Got D3D11 texture handle." << std::endl;
}
                            }
                        }
                    }
                }
                av_frame_free(&selectedFrame);
                frameProfiler.MarkStageEnd(prof::Stage::Pipeline);
                frameProfiler.EndFrameAndWrite();
                frameProfiler.MarkStageBegin(prof::Stage::Decode); // 硬解码阶段结束
            }
        }
        av_packet_unref(&packet);
    }
    av_frame_free(&frame);
    // Flush remaining batch if new mode and partial flush enabled
    if (g_batchConfig.new_batch_mode && g_batchConfig.flush_partial)
    {
        RunBatchHW();
        auto pipeline_end_us = std::chrono::steady_clock::now();
        g_batchState.pipeline_us = std::chrono::duration_cast<std::chrono::microseconds>(pipeline_end_us - pipeline_start_us).count();
        prof::BatchAggregator::Get().RecordBatch(g_batchState.windowDecoded, g_batchState.windowSelected, g_batchState.decode_total_us, g_batchState.scale_total_us, g_batchState.tensor_total_us, g_batchState.inference_us, g_batchState.pipeline_us, g_batchState.prompt, g_batchState.result);
        DBG_LOGF("[Batch] idx=%llu frames=%zu decode_window=%llu selected_window=%llu scale_us=%llu tensor_us=%llu infer_us=%llu", (unsigned long long)g_batchState.batchIndex, g_batchState.cached_scaled.size(), (unsigned long long)g_batchState.windowDecoded, (unsigned long long)g_batchState.windowSelected, (unsigned long long)g_batchState.scale_total_us, (unsigned long long)g_batchState.tensor_total_us, (unsigned long long)g_batchState.inference_us);
        g_batchState.reset();
        pipeline_start_us = std::chrono::steady_clock::now();
    }
    // After decoding completes, enqueue collected CB requests and process
    if (g_commonConfig.use_cb /*&& cb_threads_started*/)
    {
        if (g_commonConfig.cb_multi_thread)
        { 
            // Flush any remaining queued CB requests before finishing
            ProcessCBQueueAsync();
            // Signal finish; statistics thread will print summary
            StopCBThreadsIfEnabled();
        }
        else if (g_commonConfig.new_multithread) {
            // Start worker and signal final pending batch (if any)
            StartNewCBProcThreadIfEnabled();
            CbLock();
            g_cb_pending_batches++;
            CbUnlock();
            CbNotifyAll();
            CbLock();
            CbWait([](){ return g_cb_enqueued_batches >= g_cb_pending_batches; });
            CbUnlock();
            StopNewCBProcThreadIfEnabled();
        }
        else { ProcessCBQueueAndReport(); }
    }
    // av_frame_free(&frame_hw);
    //  Dump profiling stats after hardware path processing
    // prof::DumpCSV(false);
    // end-to-end decode_frames (HW) time
    auto hw_total_end_tp = std::chrono::steady_clock::now();
    g_hw_pipeline_total_us = std::chrono::duration_cast<std::chrono::microseconds>(hw_total_end_tp - hw_total_start_tp).count();
}

// 释放资源
void cleanup(AVFormatContext *format_context, AVCodecContext *codec_context)
{
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);
    avformat_network_deinit();
}
// -----------------------------------------------------------------
// Core function: Convert Windows UTF-16 path to FFmpeg UTF-8 path
// -----------------------------------------------------------------
std::string WideToUtf81(const std::wstring &wstr)
{
    if (wstr.empty())
    {
        return "";
    }

    // Step 1: Get required buffer size
    int utf8_size = WideCharToMultiByte(
        CP_UTF8,            // Target encoding: UTF-8
        0,                  // Flags
        wstr.c_str(),       // Wide character string input
        (int)wstr.length(), // Input length
        NULL,               // Output buffer (NULL to calculate size)
        0,                  // Output buffer size
        NULL, NULL          // Default character and flags (unused)
    );

    if (utf8_size == 0)
    {
        // Error handling, e.g., GetLastError()
        return "";
    }

    // Step 2: Allocate buffer and perform conversion
    std::string utf8_str(utf8_size, 0); // Allocate and initialize
    WideCharToMultiByte(
        CP_UTF8,
        0,
        wstr.c_str(),
        (int)wstr.length(),
        &utf8_str[0], // Output buffer
        utf8_size,    // Output buffer size
        NULL, NULL);

    return utf8_str;
}

// -----------------------------------------------------------------
// FFmpeg usage example
// -----------------------------------------------------------------
int main4()
{
    SetConsoleOutputCP(CP_UTF8);
    // 步骤 1: 获取含有中文的 UTF-16 路径
    // 假设这是通过文件对话框或C++的std::wstring读取到的

    std::wstring input_path_w = L"D:\\test\\VLM_test_video_clips\\002-租个女婿回家见父母-E01-E05.mp4";

    // 步骤 2: 将 UTF-16 路径转换为 FFmpeg 需要的 UTF-8 路径
    std::string input_path_utf8 = WideToUtf81(input_path_w);

    if (input_path_utf8.empty())
    {
        std::cerr << "错误：路径编码转换失败。" << std::endl;
        return 1;
    }

    // 步骤 3: 打印转换后的 UTF-8 路径
    // 注意：在某些控制台中，直接打印 UTF-8 可能会显示乱码，但FFmpeg能正确接收
    std::cout << "FFmpeg 尝试打开路径 (UTF-8): " << input_path_utf8 << std::endl;

    // -------------------------------------------------------------
    // 步骤 4: 调用 avformat_open_input
    // -------------------------------------------------------------
    AVFormatContext *pFormatCtx = NULL;
    int ret = avformat_open_input(&pFormatCtx, input_path_utf8.c_str(), NULL, NULL);

    if (ret < 0)
    {
        char err_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, err_buf, sizeof(err_buf));
        std::cerr << "错误：无法打开文件 " << input_path_utf8 << ". 错误码: " << err_buf << std::endl;
        return 1;
    }

    std::cout << "成功打开文件，格式名称: " << pFormatCtx->iformat->name << std::endl;

    // 清理资源
    avformat_close_input(&pFormatCtx);

    return 0;
}
std::wstring ExtractFilenameWithoutExt(const std::wstring &fullPath)
{
    wchar_t drive[_MAX_DRIVE];
    wchar_t dir[_MAX_DIR];
    wchar_t fname[_MAX_FNAME];
    wchar_t ext[_MAX_EXT];

    // 使用 Windows CRT 函数 _wsplitpath_s 来安全地分解路径 (UTF-16)
    _wsplitpath_s(fullPath.c_str(), drive, _MAX_DRIVE, dir, _MAX_DIR, fname, _MAX_FNAME, ext, _MAX_EXT);

    // 返回文件名部分 (不含扩展名)
    return std::wstring(fname);
}

int main(int argc, char *argv[])
{
    SetConsoleOutputCP(CP_UTF8);
    // Load JSON config early (resolve --config from wide/ansi args)
    DemoConfig cfg; std::string cfgErr; bool haveCfg = false;
    std::wstring configPathW; std::string configPath;
    // Try wide-char parsing to support Unicode paths on Windows
    ParsedArgs pa{};
    ParsedArgsW paw{};
#ifdef _WIN32
    auto getExeDirW = []() -> std::wstring {
        wchar_t buf[MAX_PATH];
        DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
        if (len == 0 || len == MAX_PATH) return L".";
        std::filesystem::path p(buf);
        return p.parent_path().wstring();
    };
    int wargc = 0;
    wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv && wargc > 0)
    {
        // Pre-scan for --config to load JSON defaults
        for (int i = 0; i < wargc; ++i) {
            std::wstring a = wargv[i];
            if (a == L"--config" && (i + 1) < wargc && wargv[i + 1] && wargv[i + 1][0] != L'-') {
                configPathW = wargv[++i];
            }
            else if (a.rfind(L"--config=", 0) == 0) {
                configPathW = a.substr(9);
            }
        }
        if (configPathW.empty()) {
            std::filesystem::path p(getExeDirW());
            p /= L"config.json";
            configPathW = p.wstring();
        }
        if (LoadJSONConfigW(configPathW, cfg, cfgErr)) {
            ApplyConfig(cfg);
            haveCfg = true;
        } else {
            std::wcerr << L"[Config] Failed to load config: " << configPathW << L" (" << Utf8ToWide(cfgErr) << L")" << std::endl;
            LocalFree(wargv);
            return 1;
        }
        /*ParsedArgsW*/ paw = parseArgsW(wargc, wargv);
        if (paw.ok)
        {
            pa.ok = true;
            pa.debug = paw.debug;
            pa.use_cb = paw.use_cb;
            pa.cb_batch_size = paw.cb_batch_size;
            pa.input = WideToUtf8(paw.input); // FFmpeg expects UTF-8
            pa.mode = WideToUtf8(paw.mode);
            // outDir as wide path; build outputs later with wide
            // store UTF-8 for fallback uses; we'll keep wide separately below
            pa.outDir = WideToUtf8(paw.outDir);
            pa.configPath = WideToUtf8(paw.configPath);
        }
        else
        {
            pa = parseArgs(argc, argv);
        }
        LocalFree(wargv);
    }
    else
    {
        // ANSI path: pre-scan --config then load JSON
        for (int i = 0; i < argc; ++i) {
            std::string a = argv[i] ? argv[i] : "";
            if (a == "--config" && (i + 1) < argc && argv[i + 1] && argv[i + 1][0] != '-') {
                configPath = argv[++i];
            }
            else if (a.rfind("--config=", 0) == 0) {
                configPath = a.substr(9);
            }
        }
        if (configPath.empty()) {
            std::filesystem::path p(getExeDirW());
            p /= L"config.json";
            configPath = WideToUtf8(p.wstring());
        }
        if (LoadJSONConfig(configPath, cfg, cfgErr)) {
            ApplyConfig(cfg);
            haveCfg = true;
        } else {
            std::cerr << "[Config] Failed to load config: " << configPath << " (" << cfgErr << ")" << std::endl;
            return 1;
        }
        pa = parseArgs(argc, argv);
    }
#else
    auto getExeDirA = [argv]() -> std::string {
        std::filesystem::path p = argv && argv[0] ? std::filesystem::absolute(argv[0]) : std::filesystem::current_path();
        return p.parent_path().string();
    };
    // Non-Windows: ANSI only
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i] ? argv[i] : "";
        if (a == "--config" && (i + 1) < argc && argv[i + 1] && argv[i + 1][0] != '-') {
            configPath = argv[++i];
        }
        else if (a.rfind("--config=", 0) == 0) {
            configPath = a.substr(9);
        }
    }
    if (configPath.empty()) {
        std::filesystem::path p(getExeDirA());
        p /= "config.json";
        configPath = p.string();
    }
    if (LoadJSONConfig(configPath, cfg, cfgErr)) {
        ApplyConfig(cfg);
        haveCfg = true;
    } else {
        std::cerr << "[Config] Failed to load config: " << configPath << " (" << cfgErr << ")" << std::endl;
        return 1;
    }
    pa = parseArgs(argc, argv);
#endif
    // If CLI parsing failed, try composing required fields from config
    if (!pa.ok)
    {
        if (cfg.commonCfg.input && cfg.commonCfg.mode) {
            pa.input = cfg.commonCfg.input.value();
            pa.mode = cfg.commonCfg.mode.value();
            pa.outDir = cfg.commonCfg.out_dir.value_or(".");
            if (cfg.batch.cb_batch_size) pa.cb_batch_size = std::max(0, cfg.batch.cb_batch_size.value());
            pa.ok = true;
        }
    }
    if (!pa.ok)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <hw|sw> [out_dir] \n"
                  << "  --sel-policy=frame|time|mixed \n"
                  << "  --frame-interval=N --window-seconds=S --max-per-window=M \n"
                  << "  --min-frames-between=N --min-seconds-between=S \n"
                  << "  --max-cached=N [--keep-cache] [--debug] \n"
                  << "  --config=PATH_TO_JSON \n"
                  << "  --max_num_seqs=N \n"
                  << "  --dynamic_split_fuse=true|false" << std::endl;
        return 1;
    }
    if (pa.debug || std::getenv("VLM_DEBUG"))
    {
        g_commonConfig.debug = true;
        SetDebugMode(g_commonConfig.debug);
    }
    // Apply CLI overrides into globals immediately after parsing
	if (!pa.input.empty()) g_commonConfig.input_video_path = pa.input;
	if (!pa.mode.empty())  g_commonConfig.hw_decode = (pa.mode == "hw" || pa.mode == "HW");
    if (!pa.outDir.empty()) g_commonConfig.log_path = pa.outDir;
    bool cli_use_cb = (pa.use_cb || paw.use_cb);
    int cli_cb_batch_size = std::max(pa.cb_batch_size, paw.cb_batch_size);
    if (cli_use_cb) { g_vlmConfig.enable_continuous_batching = true; g_commonConfig.use_cb = true; }
    if (cli_cb_batch_size > 0) g_batchConfig.cb_batch_size = cli_cb_batch_size;
    // Re-apply to globals so overrides propagate consistently
    // ApplyConfig(cfg);
    DBG_LOG(std::string("[Main] DebugMode=") + (IsDebugMode() ? "ON" : "OFF"));
    // Dump loaded config sections for visibility
    std::cout << "Log DemoConfig start" << std::endl;
    LogDemoConfig(cfg);
    // Log resolved global runtime configs
    std::cout << "Log DemoConfig end" << std::endl;
    std::cout << "Log g_fsConfig/g_batchConfig/g_commonConfig/g_vlmConfig start" << std::endl;
    LogBatchConfig(g_batchConfig);
    LogFSConfig(g_fsConfig);
    LogBatchConfig(g_batchConfig);
    LogCommonConfig(g_commonConfig);
    LogVLMConfig(g_vlmConfig);
    std::cout << "Log g_fsConfig/g_batchConfig/g_commonConfig/g_vlmConfig end" << std::endl;
    // Compose runtime params from cfg (already overridden by CLI)
    std::string input_file = g_commonConfig.input_video_path;
    //g_commonConfig.hw_decode
   // std::string mode = cfg.commonCfg.mode.value_or(!pa.mode.empty() ? pa.mode : std::string("hw"));
    std::string outDir = g_commonConfig.log_path;
#ifdef _WIN32
    // Preserve a wide outDir for output file construction
    std::wstring outDirW;
    {
        int wargc2 = 0;
        wchar_t **wargv2 = CommandLineToArgvW(GetCommandLineW(), &wargc2);
        if (wargv2 && wargc2 > 0)
        {
            ParsedArgsW paw2 = parseArgsW(wargc2, wargv2);
            if (paw2.ok)
                outDirW = paw2.outDir;
            else
                outDirW = Utf8ToWide(outDir);
            LocalFree(wargv2);
        }
        else
        {
            outDirW = Utf8ToWide(outDir);
        }
    }
#endif
   // bool useSoftware = (mode == "sw" || mode == "SW");
   // bool useHardware = (mode == "hw" || mode == "HW");
    bool useHardware = g_commonConfig.hw_decode;
    bool useSoftware = !useHardware;
    // Resolve use_cb and cb_batch_size from globals (after CLI overrides)
    use_cb = g_vlmConfig.enable_continuous_batching && g_commonConfig.use_cb;
    cb_batch_size = g_batchConfig.cb_batch_size;
    DBG_LOGF(" use_cb=%d cb_batch_size=%d, useHardware = %d, useSoftware = %d", use_cb ? 1 : 0, cb_batch_size, useHardware, useSoftware);

    // Ensure output directory exists
    try
    {
#ifdef _WIN32
        std::filesystem::create_directories(std::filesystem::path(outDirW.empty() ? Utf8ToWide(outDir) : outDirW));
#else
        std::filesystem::create_directories(outDir);
#endif
    }
    catch (const std::exception &e)
    {
        std::cerr << "[PROF] Failed to create output directory '" << outDir << "': " << e.what() << std::endl;
    }

    const char *hw_device_type = "d3d11va"; // Default hardware decoding type

    init_ffmpeg();

    AVFormatContext *format_context = nullptr;
    if (avformat_open_input(&format_context, input_file.c_str(), nullptr, nullptr) < 0)
    {
        std::cerr << "Failed to open input file: " << input_file << std::endl;
        return 1;
    }

    std::cout << "成功打开文件： "<< input_file<< " 文件格式名称: " << format_context->iformat->name << std::endl;

    if (avformat_find_stream_info(format_context, nullptr) < 0)
    {
        std::cerr << "Failed to find stream information." << std::endl;
        avformat_close_input(&format_context);
        return 1;
    }

    int stream_index = find_video_stream(format_context);
    if (stream_index < 0)
    {
        std::cerr << "No video stream found." << std::endl;
        avformat_close_input(&format_context);
        return 1;
    }

    // Derive width/height and simple codec string (avc/hevc)
    int srcW = 0, srcH = 0;
    std::wstring codecSimple = L"";
    if (format_context && stream_index >= 0)
    {
        AVCodecParameters *codec_params = format_context->streams[stream_index]->codecpar;
        srcW = codec_params->width;
        srcH = codec_params->height;
        switch (codec_params->codec_id)
        {
        case AV_CODEC_ID_H264:
            codecSimple = L"avc";
            break;
        case AV_CODEC_ID_HEVC:
            codecSimple = L"hevc";
            break;
        default:
            codecSimple = L"unknown";
            break;
        }
    }

    // Use input clip name as prefix for dump files
    // std::filesystem::path inputPath(input_file);
    // std::string clipName = inputPath.stem().string();
    std::filesystem::path inputPath(paw.input);
    std::string clipName = inputPath.stem().string();
    std::wstring filename_w = ExtractFilenameWithoutExt(paw.input);
#ifdef _WIN32

    std::filesystem::path outBaseW = std::filesystem::path(outDirW.empty() ? Utf8ToWide(outDir) : outDirW);
    std::filesystem::path profPathW = outBaseW / (filename_w + L"_profile_" + (useSoftware ? L"sw" : L"hw") + L".csv");
    std::filesystem::path resultPathW = outBaseW / (filename_w + L"_results_" + (useSoftware ? L"sw" : L"hw") + L".csv");
    std::filesystem::path batchPathW = outBaseW / (filename_w + L"_batch_" + (useSoftware ? L"sw" : L"hw") + L".csv");
#else
    std::string profFile = outDir + std::string("/") + clipName + std::string("_profile_") + (useSoftware ? "sw" : "hw") + ".csv";
    std::string resultFile = outDir + std::string("/") + clipName + std::string("_results_") + (useSoftware ? "sw" : "hw") + ".csv";
    std::string batchFile = outDir + std::string("/") + clipName + std::string("_batch_") + (useSoftware ? "sw" : "hw") + ".csv";
#endif

    // Recompute output file names to include clip dimensions and codec info
    if (srcW > 0 && srcH > 0)
    {

#ifdef _WIN32
        std::wstring suffix = L"_" + codecSimple + L"_" + std::to_wstring(srcW) + L"x" + std::to_wstring(srcH);
        profPathW = outBaseW / ((filename_w + suffix + L"_profile_" + (useSoftware ? L"sw" : L"hw")) + L".csv");
        resultPathW = outBaseW / ((filename_w + suffix + L"_results_" + (useSoftware ? L"sw" : L"hw")) + L".csv");
        batchPathW = outBaseW / ((filename_w + suffix + L"_batch_" + (useSoftware ? L"sw" : L"hw")) + L".csv");
#else
        std::string suffix = std::string("_") + codecSimple + "_" + std::to_string(srcW) + "x" + std::to_string(srcH);
        profFile = outDir + std::string("/") + clipName + suffix + std::string("_profile_") + (useSoftware ? "sw" : "hw") + ".csv";
        resultFile = outDir + std::string("/") + clipName + suffix + std::string("_results_") + (useSoftware ? "sw" : "hw") + ".csv";
        batchFile = outDir + std::string("/") + clipName + suffix + std::string("_batch_") + (useSoftware ? "sw" : "hw") + ".csv";
#endif
    }
#ifdef _WIN32
    prof::FrameProfiler::SetOutputFileW(profPathW.wstring());
    DBG_LOG(std::string("[PROF] Output -> ") + WideToUtf8(profPathW.wstring()));
    SetVLMResultFileW(resultPathW.wstring());
    prof::BatchAggregator::Get().SetOutputFileW(batchPathW.wstring());
    DBG_LOG(std::string("[BATCH] Output -> ") + WideToUtf8(batchPathW.wstring()));
#else
    prof::FrameProfiler::SetOutputFile(profFile); // 单例逐帧日志与累计日志同路径
    DBG_LOG(std::string("[PROF] Output -> ") + profFile);
    SetVLMResultFile(resultFile);
    prof::BatchAggregator::Get().SetOutputFile(batchFile);
    DBG_LOG(std::string("[BATCH] Output -> ") + batchFile);
#endif
    // (debug 已通过 parseArgs 设置)
    SetVLMInputFile(input_file);

    // useSoftware 已通过参数决定
    AVCodecContext *codec_context = nullptr;
    if (useSoftware)
    {
        DBG_LOG("[Main] Using software decoding path.");
        codec_context = open_decoder_sw(format_context, stream_index);
        if (!codec_context)
        {
            avformat_close_input(&format_context);
            return 1;
        }
        auto decode_start = std::chrono::high_resolution_clock::now();
        // Reset SW totals before run
        g_sw_inference_total_us = 0;
        g_sw_pipeline_total_us = 0;
        decode_frames_sw(format_context, codec_context, stream_index);
        auto decode_end = std::chrono::high_resolution_clock::now();

        std::cout << "total time of entire pipeline: " << std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count() << " ms" << std::endl;
        // Report SW aggregated totals
        std::cout << "[SW] total inference time: " << (g_sw_inference_total_us / 1000) << " ms" << std::endl;
        std::cout << "[SW] total decode_frames_sw time: " << (g_sw_pipeline_total_us / 1000) << " ms" << std::endl;
    }
    else
    {
        DBG_LOG(std::string("[Main] Using hardware decoding path: ") + hw_device_type);
        codec_context = open_decoder(format_context, stream_index, hw_device_type);
        if (!codec_context)
        {
            avformat_close_input(&format_context);
            return 1;
        }
        auto decode_start = std::chrono::high_resolution_clock::now();
        // Reset HW totals before run
        g_hw_inference_total_us = 0;
        g_hw_pipeline_total_us = 0;
        decode_frames(format_context, codec_context, stream_index);
        auto decode_end = std::chrono::high_resolution_clock::now();

      //  std::cout << "total time of entire pipeline: " << std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count() << " ms" << std::endl;
        // Report HW aggregated totals
        std::cout << "[HW] total inference time: " << (g_hw_inference_total_us / 1000) << " ms" << std::endl;
        std::cout << "[HW] total decode_frames time: " << (g_hw_pipeline_total_us / 1000) << " ms" << std::endl;
    }

    cleanup(format_context, codec_context);

    DBG_LOG("[VLM] Result file finalized.");

    // Print global summary
    prof::Summary::Get().PrintSummary();

    return 0;
}


