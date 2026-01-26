#pragma once
// Declarations for continuous batching benchmark components
// Corresponds to `continuous_batching_benchmark.cpp`

#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <atomic>

#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/genai/generation_handle.hpp>
#include <openvino/genai/generation_config.hpp>
//#include "parse_options.h"

namespace CBPipeline {
class AutoStartTimer {
    const decltype(std::chrono::steady_clock::now()) m_start;
public:
    AutoStartTimer() :
        m_start(std::chrono::steady_clock::now()) {
    }

    double current_in_milli() const;
};

// Simple dataset container for prompts and sampling params
struct Dataset {
    std::vector<std::string> m_prompts;
    std::vector<ov::genai::GenerationConfig> m_sampling_params;
    std::vector<size_t> m_input_lens, m_output_lens;
    size_t m_total_input_len = 0;
    size_t m_total_output_len = 0;

    // Dataset(std::string promt, ov::genai::GenerationConfig sampling_params);
    void reserve(size_t size);
    void push_data(std::string prompt, ov::genai::GenerationConfig sampling_params);
    void push_lens(size_t input_len, size_t output_len);
    float get_average_input_len() const;
    float get_average_output_len() const;
    bool empty() const;
    size_t size() const;
};

// Wraps a GenerationHandle and tracks per-sequence timing metrics
class GenerationInfo {
public:
    // Construct with a generation handle and input length for metrics
    GenerationInfo(ov::genai::GenerationHandle generation_handle, size_t input_len);

    void update_sequence(int64_t sequence_id);
    void update(ov::genai::GenerationOutputs& outputs);

    ov::genai::GenerationOutputs read();
    bool can_read();
    bool is_finished();

    void set_inactive();
    bool is_active();

    struct GenerationMetrics {
        std::chrono::milliseconds mean_ttft{0};
        std::chrono::milliseconds mean_tpot{0};
        size_t num_output_tokens{0};
        size_t num_input_tokens{0};
    };
    GenerationMetrics get_metrics();

//private:
    // Per-sequence timing info (TTFT/TPOT) used internally by GenerationInfo
    struct SequenceInfo {
        std::chrono::milliseconds ttft{0};
        std::chrono::milliseconds cumulated_tpot{0};
        std::chrono::milliseconds mean_tpot{0};
        size_t num_output_tokens{0};

        std::chrono::steady_clock::time_point start_time{};
        std::chrono::steady_clock::time_point last_read_time{};

        explicit SequenceInfo(std::chrono::steady_clock::time_point& start_time_)
            : start_time(start_time_) {}

        void update();
    };
        ov::genai::GenerationHandle generation_handle;
    std::chrono::steady_clock::time_point start_time;
    std::unordered_map<int64_t, SequenceInfo> sequences_info;
    bool active = true;
    size_t input_len;
};

// Collects GenerationInfo objects and aggregates stats
class GenerationInfoCollector {
public:
    GenerationInfoCollector() = default;
    void set_start_time(std::chrono::steady_clock::time_point start_time);
    void add_generation(ov::genai::ContinuousBatchingPipeline* pipe,
                        Dataset* dataset,
                        size_t request_id,
                        bool is_speculative_decoding_enabled);
    void add_generation(ov::genai::ContinuousBatchingPipeline* pipe, size_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params, bool is_speculative_decoding_enabled);

    void add_generation(ov::genai::ContinuousBatchingPipeline* pipe, size_t request_id,
        std::string prompt, std::vector<ov::Tensor>& images, ov::genai::GenerationConfig sampling_params, bool is_speculative_decoding_enabled);

    //add_request(uint64_t request_id, const std::string& prompt, const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos, const ov::genai::GenerationConfig& sampling_params);
    void add_generation(ov::genai::ContinuousBatchingPipeline* pipe, size_t request_id,
        std::string prompt, std::vector<ov::Tensor>& images, std::vector<ov::Tensor>& videos, ov::genai::GenerationConfig sampling_params, bool is_speculative_decoding_enabled);
    size_t run();
    void print_statistics();
    bool isCompleted();

    std::mutex mutex;
    std::vector<GenerationInfo> generations_info;
    size_t num_finished = 0;
    std::chrono::steady_clock::time_point start_time;
};

// Adds a generation request into the pipeline and collector
void AddGeneration(ov::genai::ContinuousBatchingPipeline* pipe,
                   Dataset* dataset,
                   size_t request_id,
                   GenerationInfoCollector* generation_info_collector,
                   bool is_speculative_decoding_enabled);
void AddGeneration(ov::genai::ContinuousBatchingPipeline* pipe,
    size_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params,
    GenerationInfoCollector* generation_info_collector,
    bool is_speculative_decoding_enabled);
// Drives the pipeline step loop until finish flag becomes true
void llmEngineLoop(ov::genai::ContinuousBatchingPipeline* pipe,
                   Dataset* dataset,
                   std::atomic<bool>* finishThread);

// Reads outputs from all active generations and prints aggregated statistics
void statisticsReporter(GenerationInfoCollector* generations_info_collector,
                        int num_prompts);

} // namespace CBPipeline

ov::genai::ContinuousBatchingPipeline& GetCachedCBPipeline();
CBPipeline::GenerationInfoCollector& GetCachedGenerationInfoCollector();


