// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <cstdlib>
#include <chrono>
#include <ostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <atomic>

// #include <nlohmann/json.hpp>
// #include <cxxopts.hpp>

#include "openvino/genai/cache_eviction.hpp"
// #include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
// Declarations for Dataset/GenerationInfo and helpers
#include "continuous_batching_chat.h"
#include "../utils/util.h"

// VLMConfig g_vlmConfig;
ov::genai::ContinuousBatchingPipeline& GetCachedCBPipeline()
{
    static std::unique_ptr<ov::genai::ContinuousBatchingPipeline> g_pipeline;
    static std::once_flag g_once;
    std::call_once(g_once, []()
        {
            // Read options from JSON (falls back to defaults if missing)
            std::cout << "Loading models, creating pipelines, preparing environment..." << std::endl;
            std::cout << "path: " << g_vlmConfig.path << std::endl;
            std::cout << "device: " << g_vlmConfig.device << std::endl;
            std::cout << "schedulerConfig: \n" << g_vlmConfig.shedulerConfig.to_string() << std::endl;
            g_pipeline = std::make_unique<ov::genai::ContinuousBatchingPipeline>(g_vlmConfig.path, g_vlmConfig.shedulerConfig, g_vlmConfig.device); });


    return *g_pipeline;
}
CBPipeline::GenerationInfoCollector& GetCachedGenerationInfoCollector()
{
    static std::unique_ptr<CBPipeline::GenerationInfoCollector> g_generationInfoCollector;
    static std::once_flag g_once;
    std::call_once(g_once, []()
        {
            std::cout << "Creating GenerationInfoCollector..." << std::endl;
            g_generationInfoCollector = std::make_unique<CBPipeline::GenerationInfoCollector>(); });
    return *g_generationInfoCollector;
}
namespace CBPipeline
{

    double AutoStartTimer::current_in_milli() const
    {
        auto m_end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(m_end - m_start).count();
    }


void Dataset::reserve(const size_t size)
{
    m_prompts.reserve(size);
    m_sampling_params.reserve(size);
    m_input_lens.reserve(size);
    m_output_lens.reserve(size);
}

void Dataset::push_data(std::string prompt, ov::genai::GenerationConfig sampling_params)
{
    m_prompts.push_back(std::move(prompt));
    m_sampling_params.push_back(std::move(sampling_params));
}

void Dataset::push_lens(size_t input_len, size_t output_len)
{
    m_input_lens.push_back(input_len);
    m_output_lens.push_back(output_len);
    m_total_input_len += input_len;
    m_total_output_len += output_len;
}

float Dataset::get_average_input_len() const
{
    OPENVINO_ASSERT(!empty());
    return static_cast<float>(m_total_input_len / size());
}

float Dataset::get_average_output_len() const
{
    OPENVINO_ASSERT(!empty());
    return static_cast<float>(m_total_output_len / size());
}

bool Dataset::empty() const { return size() == 0; }
size_t Dataset::size() const { return m_prompts.size(); }

// GenerationInfo method implementations
GenerationInfo::GenerationInfo(ov::genai::GenerationHandle generation_handle_, size_t input_len_)
    : input_len(input_len_)
{
    generation_handle = std::move(generation_handle_);
    start_time = std::chrono::steady_clock::now();
}

void GenerationInfo::SequenceInfo::update()
{
    auto new_read_time = std::chrono::steady_clock::now();
    if (last_read_time.time_since_epoch() == std::chrono::milliseconds::zero())
    {
        ttft = std::chrono::duration_cast<std::chrono::milliseconds>(new_read_time - start_time);
    }
    else
    {
        cumulated_tpot += std::chrono::duration_cast<std::chrono::milliseconds>(new_read_time - last_read_time);
        mean_tpot = cumulated_tpot / num_output_tokens;
    }
    num_output_tokens++;
    last_read_time = new_read_time;
}

void GenerationInfo::update_sequence(int64_t sequence_id)
{
    if (sequences_info.find(sequence_id) == sequences_info.end())
    {
        sequences_info.emplace(sequence_id, SequenceInfo(start_time));
    }
    sequences_info.at(sequence_id).update();
}

void GenerationInfo::update(ov::genai::GenerationOutputs &outputs)
{
    for (auto const &output : outputs)
    {
        update_sequence(output.first);
    }
}

ov::genai::GenerationOutputs GenerationInfo::read() { return generation_handle->read(); }
bool GenerationInfo::can_read() { return generation_handle->can_read(); }
bool GenerationInfo::is_finished() { return generation_handle->get_status() == ov::genai::GenerationStatus::FINISHED; }
void GenerationInfo::set_inactive() { active = false; }
bool GenerationInfo::is_active() { return active; }

GenerationInfo::GenerationMetrics GenerationInfo::get_metrics()
{
    GenerationMetrics generation_metrics;
    if (!sequences_info.empty())
    {
        for (auto &pair : sequences_info)
        {
            generation_metrics.mean_ttft += pair.second.ttft;
            generation_metrics.mean_tpot += pair.second.mean_tpot;
            generation_metrics.num_output_tokens += pair.second.num_output_tokens;
        }
        generation_metrics.mean_ttft /= sequences_info.size();
        generation_metrics.mean_tpot /= sequences_info.size();
        generation_metrics.num_input_tokens = input_len;
    }
    return generation_metrics;
}

// GenerationInfoCollector method implementations
void GenerationInfoCollector::set_start_time(std::chrono::steady_clock::time_point start_time_)
{
    start_time = start_time_;
}

void GenerationInfoCollector::add_generation(ov::genai::ContinuousBatchingPipeline *pipe,
                                             Dataset *dataset,
                                             size_t request_id,
                                             bool is_speculative_decoding_enabled)
{
    auto sampling_params = dataset->m_sampling_params[request_id];
    if (is_speculative_decoding_enabled)
    {
        sampling_params.num_assistant_tokens = 5; // enable static speculative decoding
        // sampling_params.assistant_confidence_threshold = 0.4f; // dynamic speculative decoding
    }
    ov::genai::GenerationHandle generation_handle = pipe->add_request(request_id, dataset->m_prompts[request_id], sampling_params);
    std::lock_guard<std::mutex> lock(mutex);
    generations_info.emplace_back(std::move(generation_handle), dataset->m_input_lens[request_id]);
}
void GenerationInfoCollector::add_generation(ov::genai::ContinuousBatchingPipeline* pipe, size_t request_id,
    std::string prompt, ov::genai::GenerationConfig sampling_params, bool is_speculative_decoding_enabled)
{
    //std::vector<std::string> prompt;
    // std::vector<ov::genai::GenerationConfig> sampling_params;

    //if (is_speculative_decoding_enabled)
    //{
    //    sampling_params.num_assistant_tokens = 5; // enable static speculative decoding
    //    // sampling_params.assistant_confidence_threshold = 0.4f; // dynamic speculative decoding
    //}
    ov::genai::GenerationHandle generation_handle = pipe->add_request(request_id, prompt, sampling_params);
    std::lock_guard<std::mutex> lock(mutex);
    generations_info.emplace_back(std::move(generation_handle), 0);
}
void GenerationInfoCollector::add_generation(ov::genai::ContinuousBatchingPipeline *pipe, size_t request_id,
    std::string prompt, std::vector<ov::Tensor>& images, ov::genai::GenerationConfig sampling_params,bool is_speculative_decoding_enabled)
{
    ov::genai::GenerationHandle generation_handle = pipe->add_request(request_id, prompt, images, sampling_params);
    std::lock_guard<std::mutex> lock(mutex);
    generations_info.emplace_back(std::move(generation_handle), 0);
}
void GenerationInfoCollector::add_generation(ov::genai::ContinuousBatchingPipeline* pipe, size_t request_id,
    std::string prompt, std::vector<ov::Tensor>& images, std::vector<ov::Tensor>& videos, ov::genai::GenerationConfig sampling_params, bool is_speculative_decoding_enabled)
{
    ov::genai::GenerationHandle generation_handle = pipe->add_request(request_id, prompt, images, videos, sampling_params);
    std::lock_guard<std::mutex> lock(mutex);
    generations_info.emplace_back(std::move(generation_handle), 0);
}

size_t GenerationInfoCollector::run()
{
    std::lock_guard<std::mutex> lock(mutex);
    for (GenerationInfo &generation_info : generations_info)
    {
        if (!generation_info.is_active())
            continue;

        if (generation_info.is_finished())
        {
            num_finished++;
            generation_info.set_inactive();
        }
        else if (generation_info.can_read())
        {
            auto outputs = generation_info.read();
            generation_info.update(outputs);
        }
    }
    return num_finished;
}

bool GenerationInfoCollector::isCompleted()
{
    std::lock_guard<std::mutex> lock(mutex);
    return num_finished == generations_info.size();
}
void GenerationInfoCollector::print_statistics()
{
    std::chrono::seconds total_duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time);
    std::chrono::milliseconds mean_ttft = std::chrono::milliseconds::zero();
    std::chrono::milliseconds mean_tpot = std::chrono::milliseconds::zero();
    size_t total_input_len = 0;
    size_t total_output_len = 0;

    for (GenerationInfo &generation_info : generations_info)
    {
        auto generation_metrics = generation_info.get_metrics();
        mean_ttft += generation_metrics.mean_ttft;
        mean_tpot += generation_metrics.mean_tpot;
        total_input_len += generation_metrics.num_input_tokens;
        total_output_len += generation_metrics.num_output_tokens;
    }
    if (!generations_info.empty())
    {
        mean_ttft /= generations_info.size();
        mean_tpot /= generations_info.size();
    }
    std::cout << "Benchmark duration: " << total_duration.count() << " s" << std::endl;
    std::cout << "Total number of input tokens: " << total_input_len << std::endl;
    std::cout << "Total number of output tokens: " << total_output_len << std::endl;
    std::cout << "Input throughput: " << (total_duration.count() ? total_input_len / total_duration.count() : 0) << " tokens / s" << std::endl;
    std::cout << "Output throughput: " << (total_duration.count() ? total_output_len / total_duration.count() : 0) << " tokens / s" << std::endl;
    std::cout << "Mean TTFT: " << mean_ttft.count() << " ms" << std::endl;
    std::cout << "Mean TPOT: " << mean_tpot.count() << " ms" << std::endl;
}
void CBPipeline::AddGeneration(ov::genai::ContinuousBatchingPipeline* pipe, Dataset* dataset, size_t request_id, GenerationInfoCollector* generation_info_collector, bool is_speculative_decoding_enabled) {

    /*
    std::cout << "Total input tokens: " << dataset->m_total_input_len << std::endl;
    std::cout << "Total output tokens: " << dataset->m_total_output_len << std::endl;
    std::cout << "Average input len: " << dataset->get_average_input_len() << " tokens" << std::endl;
    std::cout << "Average output len: " << dataset->get_average_output_len() << " tokens" << std::endl;
    */

    std::cout << "Launching traffic simulator thread with request_id: " << request_id << std::endl;
    generation_info_collector->set_start_time(std::chrono::steady_clock::now());

        std::cout << "Traffic thread adding request to the queue..." << std::endl;
        generation_info_collector->add_generation(pipe, dataset, request_id, is_speculative_decoding_enabled);

    std::cout << "All requests sent, traffic simulation finished. Exiting thread." << std::endl;
}

void CBPipeline::AddGeneration(ov::genai::ContinuousBatchingPipeline * pipe,
    size_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params,
        GenerationInfoCollector * generation_info_collector,
        bool is_speculative_decoding_enabled){
    /*
    std::cout << "Total input tokens: " << dataset->m_total_input_len << std::endl;
    std::cout << "Total output tokens: " << dataset->m_total_output_len << std::endl;
    std::cout << "Average input len: " << dataset->get_average_input_len() << " tokens" << std::endl;
    std::cout << "Average output len: " << dataset->get_average_output_len() << " tokens" << std::endl;
    */

    std::cout << "Launching traffic simulator thread with request_id: " << request_id << std::endl;
    generation_info_collector->set_start_time(std::chrono::steady_clock::now());

    std::cout << "Traffic thread adding request to the queue..." << std::endl;
    generation_info_collector->add_generation(pipe, request_id, prompt, sampling_params,  is_speculative_decoding_enabled);

    std::cout << "All requests sent, traffic simulation finished. Exiting thread." << std::endl;
}

void CBPipeline::llmEngineLoop(ov::genai::ContinuousBatchingPipeline* pipe, Dataset* dataset, std::atomic<bool>* finishThread) {
    std::cout << "Launching LLM engine thread" << std::endl;
    size_t num_finished = 0;

    while (!(*finishThread)) {
        while (pipe->has_non_finished_requests()) {
            pipe->step();
        }
    }
    std::cout << "All requests processed, LLM Engine loop escaped. Exiting thread." << std::endl;
}

void CBPipeline::statisticsReporter(GenerationInfoCollector *generations_info_collector, int num_prompts)
{
    /*int num_finished = 0;*/
    while (!generations_info_collector->isCompleted())
    {
        /*num_finished =*/ generations_info_collector->run();
    }
    std::cout << "Benchmark finished, summarizing statistics..." << std::endl;
    generations_info_collector->print_statistics();

    std::cout << "Exiting statistics reporter thread." << std::endl;
}
} // namespace

