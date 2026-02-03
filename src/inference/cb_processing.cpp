//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "cb_processing.h"
#include "continuous_batching_chat.h"
#include "../batch/batch_state.h"
#include "../video/segment_timing.h"
#include "../utils/util.h"
#include "../utils/debug.h"
#include "../utils/profiling.h"

#include <iostream>
#include <vector>

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
            std::vector<CBInferenceParams> local_queue;
            local_queue.swap(g_cbInferenceQueue);
            CbUnlock();

            if (cb_timer) {
                g_hw_cb_start = std::chrono::steady_clock::now();
                cb_timer = false;
            }

            int req_i = 0;
            std::vector<ov::Tensor> empty_images;
            for (auto& req : local_queue) {
                DBG_LOG(std::string("[CB new-mt] Enqueue request batchIndex=") + std::to_string(req.batchIndex));
                generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, empty_images, req.tensors, req.sampling_params, false);
                ++req_i;
            }
            CbLock();
            g_cb_enqueued_batches = g_cb_pending_batches;
            CbUnlock();
            CbNotifyAll();

            while (pipe.has_non_finished_requests()) {
                pipe.step();
            }

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
                    if (req.seg_has_pts)
                        RecordJsonSegmentFromTimes(req.seg_start_sec, req.seg_end_sec, combined);
                    else
                        RecordJsonSegmentFromTimes(0.0, 0.0, combined);
                }
                ++i;
            }
            generation_info_collector.generations_info.clear();
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

void ProcessCBQueueAndReport()
{
    auto cb_start = std::chrono::high_resolution_clock::now();
    auto& pipe = GetCachedCBPipeline();
    auto& generation_info_collector = GetCachedGenerationInfoCollector();
    generation_info_collector.set_start_time(std::chrono::steady_clock::now());

    int req_i = 0;
    std::cout << "Processing " << g_batchConfig.cb_batch_size << " CB requests..." << std::endl;
    std::vector<ov::Tensor> empty_images;

    if (cb_batch_size == 1)
    {
        for (auto& req : g_cbInferenceQueue)
        {
            DBG_LOG(std::string("Enqueue CB request batchIndex=") + std::to_string(req.batchIndex));
            generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, empty_images, req.tensors, req.sampling_params, false);
            auto handle = generation_info_collector.generations_info[req_i].generation_handle;
            while (handle->get_status() != ov::genai::GenerationStatus::FINISHED)
                pipe.step();
            req_i++;
        }
    }
    else
    {
        for (auto& req : g_cbInferenceQueue)
        {
            DBG_LOG(std::string("Enqueue CB request batchIndex=") + std::to_string(req.batchIndex));
            generation_info_collector.add_generation(&pipe, req.batchIndex, req.prompt, empty_images, req.tensors, req.sampling_params, false);
            req_i++;
        }
        while (pipe.has_non_finished_requests())
        {
            pipe.step();
        }
    }

    while (pipe.has_non_finished_requests())
    {
        pipe.step();
    }
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
            if (req.seg_has_pts)
                RecordJsonSegmentFromTimes(req.seg_start_sec, req.seg_end_sec, combined);
            else
                RecordJsonSegmentFromTimes(0.0, 0.0, combined);
        }
        ++i;
    }
    generation_info_collector.generations_info.clear();
    g_cbInferenceQueue.clear();
    g_cbInferenceQueue.shrink_to_fit();
}
