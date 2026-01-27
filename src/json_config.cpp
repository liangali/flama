#include "json_config.hpp"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "debug.h"

// External symbols we will update
#include "continuous_batching_chat.h"
#include "util.h"
using nlohmann::json;

static std::string ReadFileUtf8(const std::string& path, bool& ok) {
    ok = false;
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return {};
    std::ostringstream ss;
    ss << fin.rdbuf();
    ok = true;
    return ss.str();
}

#ifdef _WIN32
static std::string ReadFileWide(const std::wstring& path, bool& ok) {
    ok = false;
    FILE* f = _wfopen(path.c_str(), L"rb");
    if (!f) return {};
    std::vector<char> buf;
    char tmp[4096];
    size_t n;
    while ((n = fread(tmp, 1, sizeof(tmp), f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    fclose(f);
    ok = true;
    return std::string(buf.begin(), buf.end());
}
#endif

static void MapPolicyName(const std::string& name) {
    using FSP = FSPolicy;
    if (name == "frame") g_fsConfig.policy = FSP::FrameInterval;
    else if (name == "time") g_fsConfig.policy = FSP::TimeWindowQuota;
    else if (name == "key") g_fsConfig.policy = FSP::KeyframePriority;
    else if (name == "mixed-key") g_fsConfig.policy = FSP::MixedKeyframe;
    else g_fsConfig.policy = FSP::Mixed;
}

static void ParseInto(const json& j, DemoConfig& out) {
    // New schema: top-level common and frame_selector (preferred; overrides genai.*)
    if (j.contains("common") && j["common"].is_object()) {
        const auto& c = j["common"];
        if (c.contains("debug") && c["debug"].is_boolean()) out.commonCfg.debug = c["debug"].get<bool>();
        if (c.contains("use_cb") && c["use_cb"].is_boolean()) out.commonCfg.use_cb = c["use_cb"].get<bool>();
        if (c.contains("cb-multi-thread") && c["cb-multi-thread"].is_boolean()) out.commonCfg.cb_multi_thread = c["cb-multi-thread"].get<bool>();
        if (c.contains("new-multithread") && c["new-multithread"].is_boolean()) out.commonCfg.new_multithread = c["new-multithread"].get<bool>();
        if (c.contains("input_video_path") && c["input_video_path"].is_string()) out.commonCfg.input = c["input_video_path"].get<std::string>();
        if (c.contains("decode_mode") && c["decode_mode"].is_string()) {
            auto dm = c["decode_mode"].get<std::string>();
            if (dm == "software" || dm == "sw") out.commonCfg.mode = std::optional<std::string>{"sw"};
            else out.commonCfg.mode = std::optional<std::string>{"hw"};
        }
        if (c.contains("new_batch_mode") && c["new_batch_mode"].is_boolean()) out.batch.new_batch_mode = c["new_batch_mode"].get<bool>();
        if (c.contains("vpp_downscaling") && c["vpp_downscaling"].is_object()) {
            const auto& vpp = c["vpp_downscaling"];
            if (vpp.contains("width") && vpp["width"].is_number_integer()) out.commonCfg.vpp_width = vpp["width"].get<int>();
            if (vpp.contains("height") && vpp["height"].is_number_integer()) out.commonCfg.vpp_height = vpp["height"].get<int>();
        }
    }
        // Preferred new schema: top-level batch
        if (j.contains("batch") && j["batch"].is_object()) {
            const auto& b = j["batch"];
            if (b.contains("new_batch_mode") && b["new_batch_mode"].is_boolean()) out.batch.new_batch_mode = b["new_batch_mode"].get<bool>();
            if (b.contains("batch_trigger") && b["batch_trigger"].is_number_integer()) out.batch.batch_trigger = b["batch_trigger"].get<int>();
            if (b.contains("max_cache") && b["max_cache"].is_number_integer()) out.batch.max_cache = b["max_cache"].get<int>();
            if (b.contains("decode_window") && b["decode_window"].is_number_integer()) out.batch.decode_window = b["decode_window"].get<int>();
            if (b.contains("flush_partial") && b["flush_partial"].is_boolean()) out.batch.flush_partial = b["flush_partial"].get<bool>();
            if (b.contains("cb_batch_size") && b["cb_batch_size"].is_number_integer()) out.batch.cb_batch_size = b["cb_batch_size"].get<int>();
            if (b.contains("max_frames_per_request") && b["max_frames_per_request"].is_number_integer()) out.batch.max_frames_per_request = b["max_frames_per_request"].get<int>();
        }

    if (j.contains("frame_selector") && j["frame_selector"].is_object()) {
        const auto& fs = j["frame_selector"];
        if (fs.contains("policy") && fs["policy"].is_string()) out.frame_selector.policy = fs["policy"].get<std::string>();
        if (fs.contains("frame_interval") && fs["frame_interval"].is_number_integer()) out.frame_selector.frame_interval = fs["frame_interval"].get<int>();
        if (fs.contains("min_frames_between") && fs["min_frames_between"].is_number_integer()) out.frame_selector.min_frames_between = fs["min_frames_between"].get<int>();
        if (fs.contains("window_seconds") && fs["window_seconds"].is_number()) out.frame_selector.window_seconds = fs["window_seconds"].get<double>();
        if (fs.contains("max_per_window") && fs["max_per_window"].is_number_integer()) out.frame_selector.max_per_window = fs["max_per_window"].get<int>();
        if (fs.contains("min_seconds_between") && fs["min_seconds_between"].is_number()) out.frame_selector.min_seconds_between = fs["min_seconds_between"].get<double>();
        if (fs.contains("max_cached") && fs["max_cached"].is_number_integer()) out.frame_selector.max_cached = fs["max_cached"].get<int>();
        if (fs.contains("remove_after_process") && fs["remove_after_process"].is_boolean()) out.frame_selector.remove_after_process = fs["remove_after_process"].get<bool>();
        // Keyframe / SceneCut related
        if (fs.contains("force_keyframe") && fs["force_keyframe"].is_boolean()) out.frame_selector.force_keyframe = fs["force_keyframe"].get<bool>();
        if (fs.contains("enable_scene_cut") && fs["enable_scene_cut"].is_boolean()) out.frame_selector.enable_scene_cut = fs["enable_scene_cut"].get<bool>();
        if (fs.contains("min_frames_between_scene_cut") && fs["min_frames_between_scene_cut"].is_number_integer()) out.frame_selector.min_frames_between_scene_cut = fs["min_frames_between_scene_cut"].get<int>();
        if (fs.contains("min_seconds_between_scene_cut") && fs["min_seconds_between_scene_cut"].is_number()) out.frame_selector.min_seconds_between_scene_cut = fs["min_seconds_between_scene_cut"].get<double>();
        if (fs.contains("enable_cache") && fs["enable_cache"].is_boolean()) out.frame_selector.enable_cache = fs["enable_cache"].get<bool>();
    }

    // genai
    if (j.contains("genai") && j["genai"].is_object()) {
        const auto& g = j["genai"];
        if (g.contains("model_path") && g["model_path"].is_string()) out.vlm.model_path = g["model_path"].get<std::string>();
        if (g.contains("device") && g["device"].is_string()) out.vlm.device = g["device"].get<std::string>();
        if (g.contains("scheduler") && g["scheduler"].is_object()) {
            const auto& s = g["scheduler"];
            if (s.contains("max_num_batched_tokens") && s["max_num_batched_tokens"].is_number_integer()) out.vlm.scheduler.max_num_batched_tokens = s["max_num_batched_tokens"].get<int>();
            if (s.contains("dynamic_split_fuse") && s["dynamic_split_fuse"].is_boolean()) out.vlm.scheduler.dynamic_split_fuse = s["dynamic_split_fuse"].get<bool>();
            if (s.contains("max_num_seqs") && s["max_num_seqs"].is_number_integer()) out.vlm.scheduler.max_num_seqs = s["max_num_seqs"].get<int>();
            if (s.contains("num_kv_blocks") && s["num_kv_blocks"].is_number_integer()) out.vlm.scheduler.num_kv_blocks = s["num_kv_blocks"].get<int>();
            if (s.contains("cache_size") && s["cache_size"].is_number_integer()) out.vlm.scheduler.cache_size = s["cache_size"].get<int>();
            if (s.contains("use_cache_eviction") && s["use_cache_eviction"].is_boolean()) out.vlm.scheduler.use_cache_eviction = s["use_cache_eviction"].get<bool>();
            if (s.contains("enable_prefix_caching") && s["enable_prefix_caching"].is_boolean()) out.vlm.scheduler.enable_prefix_caching = s["enable_prefix_caching"].get<bool>();
            if (s.contains("use_sparse_attention") && s["use_sparse_attention"].is_boolean()) out.vlm.scheduler.use_sparse_attention = s["use_sparse_attention"].get<bool>();
        }
    }
}

bool LoadJSONConfig(const std::string& path, DemoConfig& out, std::string& err) {
    bool ok = false;
    auto content = ReadFileUtf8(path, ok);
    if (!ok) { err = "Failed to open config: " + path; return false; }
    try {
        auto j = json::parse(content);
        ParseInto(j, out);
        return true;
    } catch (const std::exception& ex) {
        err = std::string("JSON parse error: ") + ex.what();
        return false;
    }
}

#ifdef _WIN32
bool LoadJSONConfigW(const std::wstring& path, DemoConfig& out, std::string& err) {
    bool ok = false;
    auto content = ReadFileWide(path, ok);
    if (!ok) { err = "Failed to open config (wide)"; return false; }
    try {
        auto j = json::parse(content);
        ParseInto(j, out);
        return true;
    } catch (const std::exception& ex) {
        err = std::string("JSON parse error: ") + ex.what();
        return false;
    }
}
#endif

void ApplyConfig(const DemoConfig& cfg) {
    // Frame selector
    if (cfg.frame_selector.policy) MapPolicyName(*cfg.frame_selector.policy);
    if (cfg.frame_selector.frame_interval) g_fsConfig.frame_interval = std::max(1, *cfg.frame_selector.frame_interval);
    if (cfg.frame_selector.window_seconds) g_fsConfig.window_seconds = (float)*cfg.frame_selector.window_seconds;
    if (cfg.frame_selector.max_per_window) g_fsConfig.max_per_window = std::max(0, *cfg.frame_selector.max_per_window);
    if (cfg.frame_selector.min_frames_between) g_fsConfig.min_frames_between = std::max(0, *cfg.frame_selector.min_frames_between);
    if (cfg.frame_selector.min_seconds_between) g_fsConfig.min_seconds_between = (float)*cfg.frame_selector.min_seconds_between;
    if (cfg.frame_selector.max_cached) g_fsConfig.max_cached = std::max(0, *cfg.frame_selector.max_cached);
    if (cfg.frame_selector.remove_after_process) g_fsConfig.remove_after_process = *cfg.frame_selector.remove_after_process;
    if (cfg.frame_selector.force_keyframe) g_fsConfig.force_keyframe = *cfg.frame_selector.force_keyframe;
    if (cfg.frame_selector.enable_scene_cut) g_fsConfig.enable_scene_cut = *cfg.frame_selector.enable_scene_cut;
    if (cfg.frame_selector.min_frames_between_scene_cut) g_fsConfig.min_frames_between_scene_cut = std::max(0, *cfg.frame_selector.min_frames_between_scene_cut);
    if (cfg.frame_selector.min_seconds_between_scene_cut) g_fsConfig.min_seconds_between_scene_cut = (float)*cfg.frame_selector.min_seconds_between_scene_cut;
    if (cfg.frame_selector.enable_cache) g_fsConfig.enable_cache = *cfg.frame_selector.enable_cache;

    // Batch
    if (cfg.batch.new_batch_mode) g_batchConfig.new_batch_mode = *cfg.batch.new_batch_mode;
    if (cfg.batch.batch_trigger) g_batchConfig.batch_trigger = *cfg.batch.batch_trigger;
    if (cfg.batch.max_cache) g_batchConfig.max_cache = *cfg.batch.max_cache;
    if (cfg.batch.decode_window) g_batchConfig.decode_window = *cfg.batch.decode_window;
    if (cfg.batch.flush_partial) g_batchConfig.flush_partial = *cfg.batch.flush_partial;
    if (cfg.batch.max_frames_per_request) g_batchConfig.max_frames_per_request = std::max(1, *cfg.batch.max_frames_per_request);

    {
        // Batch/CB settings
        if (cfg.batch.cb_batch_size) g_batchConfig.cb_batch_size = std::max(0, *cfg.batch.cb_batch_size);
        if (cfg.commonCfg.use_cb) g_commonConfig.use_cb = *cfg.commonCfg.use_cb;
        if (cfg.commonCfg.cb_multi_thread) g_commonConfig.cb_multi_thread = *cfg.commonCfg.cb_multi_thread;
        if (cfg.commonCfg.new_multithread) g_commonConfig.new_multithread = *cfg.commonCfg.new_multithread;
    }
    // Do not set use_cb here to avoid cross-TU linkage to static; main will decide based on CLI+cb_batch_size
    {
        g_vlmConfig.device = *cfg.vlm.device;
        g_vlmConfig.path = *cfg.vlm.model_path;
        g_vlmConfig.enable_continuous_batching = *cfg.commonCfg.use_cb;
    }
    
    // Scheduler config
    {
        auto& sc = cfg.vlm.scheduler;
        if (sc.max_num_batched_tokens) g_vlmConfig.shedulerConfig.max_num_batched_tokens = (size_t)*sc.max_num_batched_tokens;
        if (sc.dynamic_split_fuse) g_vlmConfig.shedulerConfig.dynamic_split_fuse = *sc.dynamic_split_fuse;
        if (sc.max_num_seqs) g_vlmConfig.shedulerConfig.max_num_seqs = (size_t)*sc.max_num_seqs;
		if (sc.num_kv_blocks) g_vlmConfig.shedulerConfig.num_kv_blocks = (size_t)*sc.num_kv_blocks;
        if (sc.cache_size) g_vlmConfig.shedulerConfig.cache_size = (size_t)*sc.cache_size;
        if (sc.use_cache_eviction) g_vlmConfig.shedulerConfig.use_cache_eviction = *sc.use_cache_eviction;
        if (sc.enable_prefix_caching) g_vlmConfig.shedulerConfig.enable_prefix_caching = (size_t)*sc.enable_prefix_caching;
        if (sc.use_sparse_attention) g_vlmConfig.shedulerConfig.use_sparse_attention = (size_t)*sc.use_sparse_attention;
        // Optional fields: only apply if your SchedulerConfig supports them; ignore otherwise.
    }

    // Main flow defaults (do not override CLI later; set pre-CLI defaults)
    if (cfg.commonCfg.debug && *cfg.commonCfg.debug) {
        g_commonConfig.debug = true;
    }
    // Map main flow input/mode into CommonConfig convenience fields
    if (cfg.commonCfg.input) {
        g_commonConfig.input_video_path = *cfg.commonCfg.input;
    }
    if (cfg.commonCfg.mode) {
        const auto& m = *cfg.commonCfg.mode;
        g_commonConfig.hw_decode = !(m == "sw");
    }
    if (cfg.commonCfg.vpp_width) {
        g_commonConfig.vpp_down_width = std::max(0, *cfg.commonCfg.vpp_width);
    }
    if (cfg.commonCfg.vpp_height) {
        g_commonConfig.vpp_down_height = std::max(0, *cfg.commonCfg.vpp_height);
    }
    // mode/input/out_dir defaults should be handled by caller (main) when composing ParsedArgs
}

// ----------------------- Debug dump helpers -----------------------
template <typename T>
static std::string OptToStr(const std::optional<T>& v) {
    if (!v.has_value()) return "<unset>";
    std::ostringstream os; os << *v; return os.str();
}
template <>
std::string OptToStr<bool>(const std::optional<bool>& v) {
    if (!v.has_value()) return "<unset>";
    return v.value() ? "true" : "false";
}
template <>
std::string OptToStr<std::string>(const std::optional<std::string>& v) {
    if (!v.has_value()) return "<unset>";
    return '"' + v.value() + '"';
}

void LogDemoConfig(const DemoConfig& cfg) {
    // Frame selector
    DBG_LOG(std::string("[CFG.frame_selector] ") +
        "policy=" + OptToStr(cfg.frame_selector.policy) +
        ", frame_interval=" + OptToStr(cfg.frame_selector.frame_interval) +
        ", window_seconds=" + OptToStr(cfg.frame_selector.window_seconds) +
        ", max_per_window=" + OptToStr(cfg.frame_selector.max_per_window) +
        ", min_frames_between=" + OptToStr(cfg.frame_selector.min_frames_between) +
        ", min_seconds_between=" + OptToStr(cfg.frame_selector.min_seconds_between) +
        ", max_cached=" + OptToStr(cfg.frame_selector.max_cached) +
        ", remove_after_process=" + OptToStr(cfg.frame_selector.remove_after_process) +
        ", force_keyframe=" + OptToStr(cfg.frame_selector.force_keyframe) +
        ", enable_scene_cut=" + OptToStr(cfg.frame_selector.enable_scene_cut) +
        ", min_frames_between_scene_cut=" + OptToStr(cfg.frame_selector.min_frames_between_scene_cut) +
        ", min_seconds_between_scene_cut=" + OptToStr(cfg.frame_selector.min_seconds_between_scene_cut) +
        ", enable_cache=" + OptToStr(cfg.frame_selector.enable_cache)
    );

    // Batch
    DBG_LOG(std::string("[CFG.batch] ") +
        "new_batch_mode=" + OptToStr(cfg.batch.new_batch_mode) +
        ", batch_trigger=" + OptToStr(cfg.batch.batch_trigger) +
        ", max_cache=" + OptToStr(cfg.batch.max_cache) +
        ", decode_window=" + OptToStr(cfg.batch.decode_window) +
        ", flush_partial=" + OptToStr(cfg.batch.flush_partial) +
        ", max_frames_per_request=" + OptToStr(cfg.batch.max_frames_per_request)
    );

    // VLM
    DBG_LOG(std::string("[CFG.vlm] ") +
        "model_path=" + OptToStr(cfg.vlm.model_path) +
        ", device=" + OptToStr(cfg.vlm.device)
    );
    DBG_LOG(std::string("[CFG.vlm.scheduler] ") +
        "max_num_batched_tokens=" + OptToStr(cfg.vlm.scheduler.max_num_batched_tokens) +
        ", dynamic_split_fuse=" + OptToStr(cfg.vlm.scheduler.dynamic_split_fuse) +
        ", max_num_seqs=" + OptToStr(cfg.vlm.scheduler.max_num_seqs) +
        ", num_kv_blocks=" + OptToStr(cfg.vlm.scheduler.num_kv_blocks) +
        ", cache_size=" + OptToStr(cfg.vlm.scheduler.cache_size) +
        ", use_cache_eviction=" + OptToStr(cfg.vlm.scheduler.use_cache_eviction) +
        ", enable_prefix_caching=" + OptToStr(cfg.vlm.scheduler.enable_prefix_caching) +
        ", use_sparse_attention=" + OptToStr(cfg.vlm.scheduler.use_sparse_attention)
    );

    // Main flow
    DBG_LOG(std::string("[CFG.common] ") +
        "mode=" + OptToStr(cfg.commonCfg.mode) +
        ", debug=" + OptToStr(cfg.commonCfg.debug) +
        ", input=" + OptToStr(cfg.commonCfg.input) +
        ", out_dir=" + OptToStr(cfg.commonCfg.out_dir) +
        ", batch_trigger=" + OptToStr(cfg.commonCfg.batch_trigger) +
        ", cb_batch_size=" + OptToStr(cfg.commonCfg.cb_batch_size) +
        ", cb_multi_thread=" + OptToStr(cfg.commonCfg.cb_multi_thread) +
        ", new_multithread=" + OptToStr(cfg.commonCfg.new_multithread)
    );
}
