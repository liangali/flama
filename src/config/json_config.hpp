#pragma once
#include <string>
#include <optional>

// Forward declarations to avoid heavy includes here
struct FSConfig;
struct BatchConfig;
namespace ov { namespace genai { struct SchedulerConfig; } }

struct FrameSelectorCfg {
    // Policy selection: use either name or integer id
    std::optional<std::string> policy; // "frame"|"time"|"mixed"|"key"|"mixed-key"
    // Sample every N frames
    std::optional<int> frame_interval;
    // Time window length (seconds)
    std::optional<double> window_seconds;
    // Max frames per window
    std::optional<int> max_per_window;
    // Min frame interval between selections
    std::optional<int> min_frames_between;
    // Min time interval between selections (seconds)
    std::optional<double> min_seconds_between;
    // Max cache size
    std::optional<int> max_cached;
    // Remove cache after FetchBatch
    std::optional<bool> remove_after_process;

    // Force keyframe selection
    std::optional<bool> force_keyframe;
    // Enable scene cut frame selection
    std::optional<bool> enable_scene_cut;
    // Debounce: min frame interval for scene cuts
    std::optional<int> min_frames_between_scene_cut;
    // Debounce: min time interval for scene cuts (seconds)
    std::optional<double> min_seconds_between_scene_cut;
    // Persistent cache for selected frames (default off to avoid conflicts with external deallocation)
    std::optional<bool> enable_cache;
};

struct BatchCfg {
    std::optional<bool> new_batch_mode;
    std::optional<bool> use_cb;
    std::optional<int> batch_trigger;
    std::optional<int> cb_batch_size;
    std::optional<int> max_cache;
    std::optional<int> decode_window;
    std::optional<bool> flush_partial;
    std::optional<int> max_frames_per_request;
};

struct SchedulerCfg {
    std::optional<int> max_num_batched_tokens;
    std::optional<bool> dynamic_split_fuse;
    std::optional<int> max_num_seqs;
    std::optional<int> num_kv_blocks;
    std::optional<int> cache_size;
    std::optional<bool> use_cache_eviction;
    std::optional<bool> enable_prefix_caching;
    std::optional<bool> use_sparse_attention;
};

struct VlmCfg {
    std::optional<std::string> model_path;
    std::optional<std::string> device;
    SchedulerCfg scheduler;
};

struct CommonCfg {
    std::optional<std::string> mode;     // "hw"|"sw"
    std::optional<bool> debug;
    std::optional<bool> use_cb;
    std::optional<bool> cb_multi_thread; // "cb-multi-thread" in JSON
    std::optional<bool> new_multithread; // "new_multithread" in JSON
    std::optional<std::string> input;
    std::optional<int> batch_trigger;
    std::optional<int> cb_batch_size;
    // Optional VPP downscaling size
    std::optional<int> vpp_width;
    std::optional<int> vpp_height;
};

struct DemoConfig {
    FrameSelectorCfg frame_selector;
    BatchCfg batch;
    VlmCfg vlm;
    CommonCfg commonCfg;
};

// Load JSON from UTF-8 path
bool LoadJSONConfig(const std::string& path, DemoConfig& out, std::string& err);

#ifdef _WIN32
// Load JSON from UTF-16 path (Windows)
bool LoadJSONConfigW(const std::wstring& path, DemoConfig& out, std::string& err);
#endif

// Apply loaded config into existing globals and runtime defaults.
// - fs -> g_fsConfig
// - batch -> g_batchConfig
// - vlm.use_cb/cb_batch_size -> use_cb/cb_batch_size
// - vlm.cb_config_json -> SetCBConfigPath
// - vlm.scheduler -> shedulerConfig (where applicable)
// - commonCfg.{mode,input,debug} set defaults if CLI doesn't override later
void ApplyConfig(const DemoConfig& cfg);

// Debug log helper: dumps which fields are set in each section
void LogDemoConfig(const DemoConfig& cfg);
