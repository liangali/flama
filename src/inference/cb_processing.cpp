//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "cb_processing.h"
#include "continuous_batching_chat.h"
#include "vlm_chat.h"
#include "batch/batch_state.h"
#include "video/segment_timing.h"
#include "utils/util.h"
#include "utils/debug.h"
#include "utils/profiling.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifndef _WIN32
#include <mutex>
#include <condition_variable>
#endif

// Global thread instances
std::thread g_lmmEngineThread;
std::thread g_statisticsReporterThread;
std::thread g_cbProcThread;

// Thread state flags
bool cb_threads_started = false;
bool new_cb_proc_running = false;
std::atomic<bool> g_cb_stop{false};
std::atomic<bool> g_cb_finish_flag{false};
std::atomic<bool> finishGenerationThread{false};

// CB counters
int g_cb_pending_batches = 0;
int g_cb_enqueued_batches = 0;
bool use_cb = false;
int cb_batch_size = 0;
int report_idx = 0;

// Timing globals
std::chrono::steady_clock::time_point g_hw_cb_start;
std::chrono::steady_clock::time_point g_hw_cb_end;
std::chrono::steady_clock::time_point g_hw_cb_multi_start;
std::chrono::steady_clock::time_point g_hw_cb_multi_end;
bool cb_timer = true;

// Timing accumulators (shared with batch_state.cpp)
extern uint64_t g_hw_inference_total_us;

#ifdef _WIN32
CbSync::CbSync()
{
    InitializeSRWLock(&lock);
    InitializeConditionVariable(&cv);
}

CbSync& GetCbSync()
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

void CbLock() { AcquireSRWLockExclusive(&GetCbSync().lock); }
void CbUnlock() { ReleaseSRWLockExclusive(&GetCbSync().lock); }
void CbNotifyAll() { WakeAllConditionVariable(&GetCbSync().cv); }

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

void CbLock() { g_cb_mutex.lock(); }
void CbUnlock() { g_cb_mutex.unlock(); }
void CbNotifyAll() { g_cb_cv.notify_all(); }

template <typename Pred>
static void CbWait(Pred pred)
{
    std::unique_lock<std::mutex> lk(g_cb_mutex, std::adopt_lock);
    g_cb_cv.wait(lk, pred);
    lk.release();
}
#endif

namespace
{
struct CBBatchRunResult
{
    std::vector<std::string> outputs;
    uint64_t inference_us = 0;
};

ov::Tensor PackFramesAsSingleVideoTensor(const std::vector<ov::Tensor>& frames)
{
    if (frames.empty())
    {
        throw std::runtime_error("PackFramesAsSingleVideoTensor: empty frame list");
    }

    const ov::Shape firstShape = frames.front().get_shape();
    size_t h = 0;
    size_t w = 0;
    size_t c = 0;
    if (firstShape.size() == 4)
    {
        if (firstShape[0] != 1)
            throw std::runtime_error("PackFramesAsSingleVideoTensor: frame tensor N must be 1");
        h = firstShape[1];
        w = firstShape[2];
        c = firstShape[3];
    }
    else if (firstShape.size() == 3)
    {
        h = firstShape[0];
        w = firstShape[1];
        c = firstShape[2];
    }
    else
    {
        throw std::runtime_error("PackFramesAsSingleVideoTensor: unsupported frame tensor shape");
    }

    if (c != 3)
        throw std::runtime_error("PackFramesAsSingleVideoTensor: expected RGB channels == 3");

    const size_t frameBytes = h * w * c;
    const size_t packedFrameCount = std::max<size_t>(frames.size(), 4);
    ov::Tensor packed(ov::element::u8, ov::Shape{packedFrameCount, h, w, c});
    auto* dst = packed.data<uint8_t>();

    for (size_t i = 0; i < packedFrameCount; ++i)
    {
        const ov::Tensor& srcFrame = frames[std::min(i, frames.size() - 1)];
        const ov::Shape s = srcFrame.get_shape();
        bool same = false;
        if (s.size() == 4)
            same = (s[0] == 1 && s[1] == h && s[2] == w && s[3] == c);
        else if (s.size() == 3)
            same = (s[0] == h && s[1] == w && s[2] == c);
        if (!same)
            throw std::runtime_error("PackFramesAsSingleVideoTensor: inconsistent frame shape");

        std::memcpy(dst + i * frameBytes, srcFrame.data<const uint8_t>(), frameBytes);
    }

    return packed;
}

CBBatchRunResult RunCBBatchRequests(std::vector<CBInferenceParams>& requests)
{
    CBBatchRunResult runResult;
    if (requests.empty())
    {
        return runResult;
    }

    auto& pipe = GetCachedCBPipeline();
    std::vector<std::string> prompts;
    std::vector<std::vector<ov::Tensor>> images;
    std::vector<std::vector<ov::Tensor>> videos;
    std::vector<ov::genai::GenerationConfig> samplingParams;
    prompts.reserve(requests.size());
    images.reserve(requests.size());
    videos.reserve(requests.size());
    samplingParams.reserve(requests.size());

    for (auto& req : requests)
    {
        prompts.push_back(req.prompt);
        samplingParams.push_back(req.sampling_params);

        std::vector<ov::Tensor> imageInputs;
        std::vector<ov::Tensor> videoInputs;
        if (req.use_video_input)
        {
            ov::Tensor packedVideo = PackFramesAsSingleVideoTensor(req.tensors);
            videoInputs.emplace_back(std::move(packedVideo));
        }
        else
        {
            imageInputs = std::move(req.tensors);
        }
        images.push_back(std::move(imageInputs));
        videos.push_back(std::move(videoInputs));
    }

    const auto start = std::chrono::high_resolution_clock::now();
    auto results = pipe.generate(prompts, images, videos, samplingParams);
    const auto end = std::chrono::high_resolution_clock::now();
    runResult.inference_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    auto tokenizer = pipe.get_tokenizer();
    uint64_t totalInputTokens = 0;
    uint64_t totalOutputTokens = 0;
    uint64_t visibleOutputTokens = 0;
    std::vector<uint64_t> outputMetricValues;
    outputMetricValues.reserve(results.size());

    runResult.outputs.reserve(results.size());
    for (auto& result : results)
    {
        auto& metrics = result.perf_metrics;
        totalInputTokens += metrics.get_num_input_tokens();

        const uint64_t outputMetric = metrics.get_num_generated_tokens();
        totalOutputTokens += outputMetric;
        outputMetricValues.push_back(outputMetric);

        std::string text = static_cast<std::string>(result);
        runResult.outputs.push_back(text);

        const auto encoded = tokenizer.encode(text);
        const auto& shape = encoded.input_ids.get_shape();
        if (!shape.empty())
        {
            visibleOutputTokens += shape.back();
        }
    }

    uint64_t finalOutputTokens = totalOutputTokens;
    if (outputMetricValues.size() > 1)
    {
        const uint64_t firstOutputMetric = outputMetricValues.front();
        const bool allSame = std::all_of(
            outputMetricValues.begin() + 1,
            outputMetricValues.end(),
            [firstOutputMetric](uint64_t value) { return value == firstOutputMetric; });

        if (allSame)
        {
            // OpenVINO batched VLM currently reports batch-level generated token count
            // in each VLMDecodedResults item. Detect that case and count it once.
            const uint64_t specialTokenAllowance = static_cast<uint64_t>(outputMetricValues.size()) * 4;
            if (firstOutputMetric >= visibleOutputTokens &&
                firstOutputMetric <= visibleOutputTokens + specialTokenAllowance)
            {
                finalOutputTokens = firstOutputMetric;
            }
        }
    }

    AddVLMTokenTotalsWithRequestCount(
        totalInputTokens,
        finalOutputTokens,
        static_cast<uint64_t>(results.size()));

    return runResult;
}
}

void CBLmmEngineThreadFunc()
{
    auto& collector = GetCachedGenerationInfoCollector();
    while (!g_cb_finish_flag.load())
    {
        collector.run();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void CBStatisticsThreadFunc()
{
    while (!g_cb_finish_flag.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    ReportCBQueueAsync();
}

void StartNewCBProcThreadIfEnabled()
{
    if (!g_commonConfig.use_cb) return;
    if (!g_commonConfig.new_multithread) return;
    if (g_commonConfig.cb_multi_thread) return;
    if (new_cb_proc_running) return;
    std::cout << "[CB] Starting new-multithread (ProcessCBQueueAndReport)..." << std::endl;
    std::cout << "[CB] Mode: new_multithread (pending/enqueued handshake active)" << std::endl;
    g_cb_stop.store(false);
    g_hw_cb_multi_start = std::chrono::steady_clock::now();
    g_cbProcThread = std::thread([]() {
        while (true) {
            CbLock();
            CbWait([]() { return g_cb_stop.load() || (g_cb_pending_batches > g_cb_enqueued_batches); });
            if (g_cb_stop.load() && !(g_cb_pending_batches > g_cb_enqueued_batches)) {
                CbUnlock();
                break;
            }
            std::vector<CBInferenceParams> local_queue;
            local_queue.swap(g_cbInferenceQueue);
            CbUnlock();

            if (cb_timer) {
                g_hw_cb_start = std::chrono::steady_clock::now();
                cb_timer = false;
            }

            CbLock();
            g_cb_enqueued_batches = g_cb_pending_batches;
            CbUnlock();
            CbNotifyAll();

            auto runResult = RunCBBatchRequests(local_queue);
            g_hw_cb_end = std::chrono::steady_clock::now();
            g_hw_inference_total_us += runResult.inference_us;

            const size_t reportCount = std::min(local_queue.size(), runResult.outputs.size());
            for (size_t i = 0; i < reportCount; ++i) {
                const auto& req = local_queue[i];
                const auto& combined = runResult.outputs[i];
                DBG_LOG(std::string("[rslt_cb new-mt]") + combined);
                prof::BatchAggregator::Get().RecordBatch(
                    req.windowDecoded,
                    req.windowSelected,
                    req.decode_total_us,
                    req.scale_total_us,
                    req.tensor_total_us,
                    runResult.inference_us,
                    req.pipeline_us,
                    req.prompt,
                    combined);
                if (req.seg_has_pts)
                    RecordJsonSegmentFromTimes(req.seg_start_sec, req.seg_end_sec, combined);
                else
                    RecordJsonSegmentFromTimes(0.0, 0.0, combined);
            }
        }
    });
    new_cb_proc_running = true;
}

void StopNewCBProcThreadIfEnabled()
{
    if (!new_cb_proc_running) return;
    g_cb_stop.store(true);
    CbNotifyAll();
    if (g_cbProcThread.joinable()) g_cbProcThread.join();
    new_cb_proc_running = false;
    g_hw_cb_multi_end = std::chrono::steady_clock::now();
    g_hw_inference_total_us = std::chrono::duration_cast<std::chrono::microseconds>(g_hw_cb_multi_end - g_hw_cb_multi_start).count();
    std::cout << "[CB] new-multithread finished. total inference time : " << g_hw_inference_total_us / 1000.0 << std::endl;
}

void StartCBThreadsIfEnabled()
{
    if (!g_commonConfig.use_cb) return;
    if (!g_commonConfig.cb_multi_thread) return;
    if (cb_threads_started) return;
    std::cout << "[CB] Starting CB engine and statistics reporter threads..." << std::endl;

    g_cb_finish_flag.store(false);
    auto& collector = GetCachedGenerationInfoCollector();
    auto& pipe = GetCachedCBPipeline();
    collector.set_start_time(std::chrono::steady_clock::now());
    g_lmmEngineThread = std::thread(CBPipeline::llmEngineLoop, &pipe, nullptr, &finishGenerationThread);
    g_statisticsReporterThread = std::thread(CBPipeline::statisticsReporter, &collector, 0);
    cb_threads_started = true;
}

void StopCBThreadsIfEnabled()
{
    if (!g_commonConfig.use_cb) return;
    if (!g_commonConfig.cb_multi_thread) return;
    if (!cb_threads_started) return;
    std::cout << "[CB] Stopping CB engine and statistics reporter threads..." << std::endl;
    g_cb_finish_flag.store(true);
    if (g_statisticsReporterThread.joinable()) g_statisticsReporterThread.join();
    finishGenerationThread = true;
    if (g_lmmEngineThread.joinable()) g_lmmEngineThread.join();
    cb_threads_started = false;
}

void ProcessCBQueueAsync()
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

    std::cout << "Processing " << g_batchConfig.cb_batch_size << " CB requests..." << std::endl;
    std::vector<ov::Tensor> empty_images;

    for (auto& req : g_cbInferenceQueue)
    {
        DBG_LOG(std::string("Enqueue CB request batchIndex=") + std::to_string(req.batchIndex));
        std::vector<ov::Tensor> videos;
        if (req.use_video_input)
        {
            videos.emplace_back(PackFramesAsSingleVideoTensor(req.tensors));
        }
        else
        {
            videos = req.tensors;
        }
        generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, empty_images, videos, req.sampling_params, false);
    }

    auto cb_end = std::chrono::high_resolution_clock::now();
    std::cout << "Number of batch:" << g_cbInferenceQueue.size() << std::endl;
    std::cout << "CB total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cb_end - cb_start).count() << " ms" << std::endl;
    StartCBThreadsIfEnabled();
    g_cbInferenceQueue.clear();
    g_cbInferenceQueue.shrink_to_fit();
}

void ReportCBQueueAsync()
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
        size_t outputTokens = 0;
        std::string combined;
        combined.reserve(1024);
        for (size_t k = 0; k < outputs.size(); ++k)
        {
            outputTokens += outputs[k].generated_ids.size();
            auto text = tokenizer.decode(outputs[k].generated_ids);
            if (!combined.empty())
                combined += "\n";
            combined += text;
        }
        AddVLMTokenTotals(generation_info.input_len, outputTokens);
        DBG_LOG(std::string("[rslt_cb of batch NO. ") + std::to_string(i + 1) + ":]\n " + combined);
        i++;
    }
    generation_info_collector.reset();
}

void ProcessCBQueueAndReport()
{
    auto cb_start = std::chrono::high_resolution_clock::now();
    auto& pipe = GetCachedCBPipeline();
    std::cout << "Processing " << g_batchConfig.cb_batch_size << " CB requests..." << std::endl;
    for (const auto& req : g_cbInferenceQueue)
    {
        DBG_LOG(std::string("Enqueue CB request batchIndex=") + std::to_string(req.batchIndex));
    }
    auto runResult = RunCBBatchRequests(g_cbInferenceQueue);
    auto cb_end = cb_start + std::chrono::microseconds(runResult.inference_us);

    ov::genai::PipelineMetrics metrics = pipe.get_metrics();
    std::cout << "CB Metrics:" << std::endl;
    std::cout << "Cache Usage: " << metrics.cache_usage << std::endl;
    std::cout << "metrics.max_cache_usage:" << metrics.max_cache_usage << std::endl;
    std::cout << "Scheduled Requests:" << metrics.scheduled_requests << std::endl;
    std::cout << "Number of batch:" << g_cbInferenceQueue.size() << std::endl;
    std::cout << "current CB total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cb_end - cb_start).count() << " ms" << std::endl;
    g_hw_inference_total_us += runResult.inference_us;
    const uint64_t cb_inference_us = runResult.inference_us;
    const size_t reportCount = std::min(g_cbInferenceQueue.size(), runResult.outputs.size());
    for (size_t i = 0; i < reportCount; ++i)
    {
        report_idx++;
        const std::string& combined = runResult.outputs[i];
        DBG_LOG(std::string("[rslt_cb of batch NO. ") + std::to_string(report_idx) + ":]\n " + combined);
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
        if (req.seg_has_pts)
            RecordJsonSegmentFromTimes(req.seg_start_sec, req.seg_end_sec, combined);
        else
            RecordJsonSegmentFromTimes(0.0, 0.0, combined);
    }
    g_cbInferenceQueue.clear();
    g_cbInferenceQueue.shrink_to_fit();
}
