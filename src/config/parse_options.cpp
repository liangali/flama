#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <filesystem>

#include "parse_options.h"
#include "json_config.hpp"
#include "../utils/util.h"
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif
namespace {
static bool ParseBoolValue(const std::string &v, bool &out)
{
    if (v == "1" || v == "true" || v == "True" || v == "TRUE" || v == "yes" || v == "on") { out = true; return true; }
    if (v == "0" || v == "false" || v == "False" || v == "FALSE" || v == "no" || v == "off") { out = false; return true; }
    return false;
}
static bool ParseBoolValueW(const std::wstring &v, bool &out)
{
    if (v == L"1" || v == L"true" || v == L"True" || v == L"TRUE" || v == L"yes" || v == L"on") { out = true; return true; }
    if (v == L"0" || v == L"false" || v == L"False" || v == L"FALSE" || v == L"no" || v == L"off") { out = false; return true; }
    return false;
}
static bool SplitArg(const std::string &a, std::string &key, std::string &val)
{
    if (a.rfind("--", 0) != 0) return false;
    size_t pos = a.find('=');
    if (pos == std::string::npos || pos <= 2) return false;
    key = a.substr(2, pos - 2);
    val = a.substr(pos + 1);
    return true;
}
static bool SplitArgW(const std::wstring &a, std::wstring &key, std::wstring &val)
{
    if (a.rfind(L"--", 0) != 0) return false;
    size_t pos = a.find(L'=');
    if (pos == std::wstring::npos || pos <= 2) return false;
    key = a.substr(2, pos - 2);
    val = a.substr(pos + 1);
    return true;
}
static bool FindArgValueA(int argc, char* argv[], const std::string& key, std::string& out)
{
    std::string prefix = "--" + key + "=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i] ? argv[i] : "";
        if (a.rfind(prefix, 0) == 0) {
            out = a.substr(prefix.size());
            return true;
        }
    }
    return false;
}
static bool HasHelpA(int argc, char* argv[])
{
    std::string key, val;
    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) continue;
        std::string a = argv[i];
        if (a == "--help" || a == "-h" || a == "/?") {
            return true;
        }
        if (SplitArg(a, key, val) && key == "help") {
            bool b = true;
            return !ParseBoolValue(val, b) || b;
        }
    }
    return false;
}
#ifdef _WIN32
static bool FindArgValueW(int argc, wchar_t* argv[], const std::wstring& key, std::wstring& out)
{
    std::wstring prefix = L"--" + key + L"=";
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i] ? argv[i] : L"";
        if (a.rfind(prefix, 0) == 0) {
            out = a.substr(prefix.size());
            return true;
        }
    }
    return false;
}
static bool HasHelpW(int argc, wchar_t* argv[])
{
    std::wstring key, val;
    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) continue;
        std::wstring a = argv[i];
        if (a == L"--help" || a == L"-h" || a == L"/?") {
            return true;
        }
        if (SplitArgW(a, key, val) && key == L"help") {
            bool b = true;
            return !ParseBoolValueW(val, b) || b;
        }
    }
    return false;
}
#endif
} // namespace
// �����н���ʵ��
ParsedArgs parseArgs(int argc, char* argv[]) {
    ParsedArgs pa;
    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) continue;
        std::string a = argv[i];
        std::string key, val;
        if (!SplitArg(a, key, val))
            continue;
        if (key == "input") {
            pa.input = val;
        }
        else if (key == "mode") {
            pa.mode = val;
        }
        else if (key == "out_dir") {
            pa.outDir = val;
        }
        else if (key == "config") {
            pa.configPath = val;
        }
        else if (key == "video_dir" || key == "video-dir") {
            pa.videoDir = val;
        }
        else if (key == "json_file" || key == "json-file") {
            pa.jsonFile = val;
        }
        else if (key == "debug") {
            bool b = false;
            if (ParseBoolValue(val, b)) { pa.debug = b; g_commonConfig.debug = b; }
        }
        else if (key == "max_num_seqs") {
            int v = std::stoi(val.c_str());
            if (v > 0) {
                pa.max_num_seqs = v; g_vlmConfig.shedulerConfig.max_num_seqs = v;
            }
        }
        else if (key == "dynamic_split_fuse") {
            bool b = true;
            if (ParseBoolValue(val, b)) {
                pa.dynamic_split_fuse = b;
                g_vlmConfig.shedulerConfig.dynamic_split_fuse = b;
            }
        }
        else if (key == "use_cb") {
            bool b = false;
            if (ParseBoolValue(val, b)) {
                pa.use_cb = b;
                g_commonConfig.use_cb = b;
                g_vlmConfig.enable_continuous_batching = b;
            }
        }
        else if (key == "new_multithread" || key == "new-multithread") {
            bool b = false;
            if (ParseBoolValue(val, b)) { pa.new_multithread = b; g_commonConfig.new_multithread = b; }
        }
        else if (key == "cb_multi_thread" || key == "cb-multi-thread") {
            bool b = false;
            if (ParseBoolValue(val, b)) { pa.cb_multi_thread = b; g_commonConfig.cb_multi_thread = b; }
        }
        else if (key == "cb_batch_size") {
            pa.cb_batch_size = std::max(0, std::stoi(val.c_str()));
            g_batchConfig.cb_batch_size = std::max(0, std::stoi(val.c_str()));
        }
        else if (key == "prompt") {
            pa.prompt = val;
            if (!pa.prompt.empty()) g_commonConfig.prompt_video = pa.prompt;
        }
        else if (key == "sel-policy") {
            std::string v = val;
            if (v == "frame") g_fsConfig.policy = FSPolicy::FrameInterval;
            else if (v == "time") g_fsConfig.policy = FSPolicy::TimeWindowQuota;
            else if (v == "key") g_fsConfig.policy = FSPolicy::KeyframePriority;
            else if (v == "mixed-key") g_fsConfig.policy = FSPolicy::MixedKeyframe;
            else g_fsConfig.policy = FSPolicy::Mixed;
        }
        else if (key == "frame-interval") {
            g_fsConfig.frame_interval = std::max(1, std::stoi(val));
        }
        else if (key == "window-seconds") {
            g_fsConfig.window_seconds = std::stod(val);
            if (g_fsConfig.window_seconds < 0.0) g_fsConfig.window_seconds = 0.0;
        }
        else if (key == "max-per-window") {
            g_fsConfig.max_per_window = std::stoi(val);
            if (g_fsConfig.max_per_window < 0) g_fsConfig.max_per_window = 0;
        }
        else if (key == "min-frames-between") {
            g_fsConfig.min_frames_between = std::stoi(val);
            if (g_fsConfig.min_frames_between < 0) g_fsConfig.min_frames_between = 0;
        }
        else if (key == "min-seconds-between") {
            g_fsConfig.min_seconds_between = std::stod(val);
            if (g_fsConfig.min_seconds_between < 0.0) g_fsConfig.min_seconds_between = 0.0;
        }
        else if (key == "max-cached") {
            //g_fsConfig.max_cached = std::stoi(a.substr(13));
            //if (g_fsConfig.max_cached < 1) g_fsConfig.max_cached = 1;
        }
        else if (key == "keep-cache") {
            bool b = false;
            if (ParseBoolValue(val, b)) g_fsConfig.remove_after_process = !b;
        }
        else if (key == "no-force-keyframe") {
            bool b = false;
            if (ParseBoolValue(val, b)) g_fsConfig.force_keyframe = !b;
        }
        else if (key == "disable-scene-cut") {
            bool b = false;
            if (ParseBoolValue(val, b)) g_fsConfig.enable_scene_cut = !b;
        }
        else if (key == "min-scene-cut-frames") {
            g_fsConfig.min_frames_between_scene_cut = std::stoi(val);
            if (g_fsConfig.min_frames_between_scene_cut < 0) g_fsConfig.min_frames_between_scene_cut = 0;
        }
        else if (key == "min-scene-cut-seconds") {
            g_fsConfig.min_seconds_between_scene_cut = std::stod(val);
            if (g_fsConfig.min_seconds_between_scene_cut < 0.0) g_fsConfig.min_seconds_between_scene_cut = 0.0;
        }
        else if (key == "enable-cache" || key == "cache") {
            bool b = false;
            if (ParseBoolValue(val, b)) g_fsConfig.enable_cache = b;
        }
        else if (key == "batch-trigger") {
            g_batchConfig.batch_trigger = std::max(1, std::stoi(val));
        }
        else if (key == "max-cache") {
            g_batchConfig.max_cache = std::max(1, std::stoi(val));
        }
        else if (key == "decode-window") {
            g_batchConfig.decode_window = std::max(0, std::stoi(val));
        }
        else if (key == "flush-partial") {
            bool b = true;
            if (ParseBoolValue(val, b)) g_batchConfig.flush_partial = b;
        }
        else if (key == "new-batch-mode") {
            bool b = true;
            if (ParseBoolValue(val, b)) g_batchConfig.new_batch_mode = b;
        }
        else if (key == "legacy-batch-mode") {
            bool b = false;
            if (ParseBoolValue(val, b)) g_batchConfig.new_batch_mode = !b;
        }
    }
    if (pa.outDir.empty()) {
        pa.outDir = ".";
    }
    pa.ok = (!pa.mode.empty() && (!pa.input.empty() || !pa.videoDir.empty()));
    return pa;
}

// ���ַ�����ʵ��
ParsedArgsW parseArgsW(int argc, wchar_t* argv[]) {
    ParsedArgsW pa;
    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) continue;
        std::wstring a = argv[i];
        std::wstring key, val;
        if (!SplitArgW(a, key, val))
            continue;
        if (key == L"input") {
            pa.input = val;
        }
        else if (key == L"mode") {
            pa.mode = val;
        }
        else if (key == L"out_dir") {
            pa.outDir = val;
        }
        else if (key == L"config") {
            pa.configPath = val;
        }
        else if (key == L"video_dir" || key == L"video-dir") {
            pa.videoDir = val;
        }
        else if (key == L"json_file" || key == L"json-file") {
            pa.jsonFile = val;
        }
        else if (key == L"debug") {
            bool b = false;
            if (ParseBoolValueW(val, b)) { pa.debug = b; g_commonConfig.debug = b; }
        }
        else if (key == L"max_num_seqs") {
            int v = _wtoi(val.c_str());
            if (v > 0) {
                pa.max_num_seqs = v; g_vlmConfig.shedulerConfig.max_num_seqs = v;
            }
        }
        else if (key == L"dynamic_split_fuse") {
            bool b = true;
            if (ParseBoolValueW(val, b)) {
                pa.dynamic_split_fuse = b;
                g_vlmConfig.shedulerConfig.dynamic_split_fuse = b;
            }
        }
        else if (key == L"use_cb") {
            bool b = false;
            if (ParseBoolValueW(val, b)) {
                pa.use_cb = b;
                g_commonConfig.use_cb = b;
                g_vlmConfig.enable_continuous_batching = b;
            }
        }
        else if (key == L"new_multithread" || key == L"new-multithread") {
            bool b = false;
            if (ParseBoolValueW(val, b)) { pa.new_multithread = b; g_commonConfig.new_multithread = b; }
        }
        else if (key == L"cb_multi_thread" || key == L"cb-multi-thread") {
            bool b = false;
            if (ParseBoolValueW(val, b)) { pa.cb_multi_thread = b; g_commonConfig.cb_multi_thread = b; }
        }
        else if (key == L"cb_batch_size") {
            pa.cb_batch_size = std::max(0, _wtoi(val.c_str()));
            g_batchConfig.cb_batch_size = std::max(0, _wtoi(val.c_str()));
        }
        else if (key == L"prompt") {
            pa.prompt = val;
            if (!pa.prompt.empty()) g_commonConfig.prompt_video = WideToUtf8(pa.prompt);
        }
        else if (key == L"sel-policy") {
            std::wstring v = val;
            if (v == L"frame") g_fsConfig.policy = FSPolicy::FrameInterval;
            else if (v == L"time") g_fsConfig.policy = FSPolicy::TimeWindowQuota;
            else if (v == L"key") g_fsConfig.policy = FSPolicy::KeyframePriority;
            else if (v == L"mixed-key") g_fsConfig.policy = FSPolicy::MixedKeyframe;
            else g_fsConfig.policy = FSPolicy::Mixed;
        }
        else if (key == L"frame-interval") {
            g_fsConfig.frame_interval = std::max(1, _wtoi(val.c_str()));
        }
        else if (key == L"window-seconds") {
            g_fsConfig.window_seconds = _wtof(val.c_str());
            if (g_fsConfig.window_seconds < 0.0) g_fsConfig.window_seconds = 0.0;
        }
        else if (key == L"max-per-window") {
            g_fsConfig.max_per_window = _wtoi(val.c_str());
            if (g_fsConfig.max_per_window < 0) g_fsConfig.max_per_window = 0;
        }
        else if (key == L"min-frames-between") {
            g_fsConfig.min_frames_between = _wtoi(val.c_str());
            if (g_fsConfig.min_frames_between < 0) g_fsConfig.min_frames_between = 0;
        }
        else if (key == L"min-seconds-between") {
            g_fsConfig.min_seconds_between = _wtof(val.c_str());
            if (g_fsConfig.min_seconds_between < 0.0) g_fsConfig.min_seconds_between = 0.0;
        }
        else if (key == L"max-cached") {
            //g_fsConfig.max_cached = _wtoi(a.substr(13).c_str());
            //if (g_fsConfig.max_cached < 1) g_fsConfig.max_cached = 1;
        }
        else if (key == L"keep-cache") {
            bool b = false;
            if (ParseBoolValueW(val, b)) g_fsConfig.remove_after_process = !b;
        }
        else if (key == L"no-force-keyframe") {
            bool b = false;
            if (ParseBoolValueW(val, b)) g_fsConfig.force_keyframe = !b;
        }
        else if (key == L"disable-scene-cut") {
            bool b = false;
            if (ParseBoolValueW(val, b)) g_fsConfig.enable_scene_cut = !b;
        }
        else if (key == L"min-scene-cut-frames") {
            g_fsConfig.min_frames_between_scene_cut = _wtoi(val.c_str());
            if (g_fsConfig.min_frames_between_scene_cut < 0) g_fsConfig.min_frames_between_scene_cut = 0;
        }
        else if (key == L"min-scene-cut-seconds") {
            g_fsConfig.min_seconds_between_scene_cut = _wtof(val.c_str());
            if (g_fsConfig.min_seconds_between_scene_cut < 0.0) g_fsConfig.min_seconds_between_scene_cut = 0.0;
        }
        else if (key == L"enable-cache" || key == L"cache") {
            bool b = false;
            if (ParseBoolValueW(val, b)) g_fsConfig.enable_cache = b;
        }
        else if (key == L"batch-trigger") {
            g_batchConfig.batch_trigger = std::max(1, _wtoi(val.c_str()));
        }
        else if (key == L"max-cache") {
            g_batchConfig.max_cache = std::max(1, _wtoi(val.c_str()));
        }
        else if (key == L"decode-window") {
            g_batchConfig.decode_window = std::max(0, _wtoi(val.c_str()));
        }
        else if (key == L"flush-partial") {
            bool b = true;
            if (ParseBoolValueW(val, b)) g_batchConfig.flush_partial = b;
        }
        else if (key == L"new-batch-mode") {
            bool b = true;
            if (ParseBoolValueW(val, b)) g_batchConfig.new_batch_mode = b;
        }
        else if (key == L"legacy-batch-mode") {
            bool b = false;
            if (ParseBoolValueW(val, b)) g_batchConfig.new_batch_mode = !b;
        }
    }
    if (pa.outDir.empty()) {
        pa.outDir = L".";
    }
    pa.ok = (!pa.mode.empty() && (!pa.input.empty() || !pa.videoDir.empty()));
    return pa;
}

void PrintHelp(const char* exeName)
{
    const char* exe = exeName ? exeName : "flama";
    std::cout
        << "Usage:\n"
        << "  " << exe << " --input=PATH --mode=hw|sw [--out_dir=PATH]\n"
        << "  " << exe << " --video_dir=PATH --mode=hw|sw [--out_dir=PATH]\n"
        << "\n"
        << "All arguments use --name=value format only.\n"
        << "\n"
        << "Core:\n"
        << "  --input=PATH           Input video file path.\n"
        << "  --video_dir=PATH       Directory of video files (sorted by filename).\n"
        << "  --mode=hw|sw           Decode path: hw (D3D11) or sw.\n"
        << "  --out_dir=PATH         Output directory for CSV logs.\n"
        << "  --config=PATH          JSON config path (default: exe_dir/config.json).\n"
        << "  --json_file=PATH       Output JSON file for VLM results (default: ./output_vlm.json).\n"
        << "  --prompt=TEXT          Override VLM prompt for video batches.\n"
        << "  --debug=0|1            Enable debug logging.\n"
        << "\n"
        << "Frame selection:\n"
        << "  --sel-policy=frame|time|mixed|key|mixed-key\n"
        << "  --frame-interval=N     Select every N frames (frame policy).\n"
        << "  --window-seconds=S     Window size in seconds (time policy).\n"
        << "  --max-per-window=M     Max selected frames per window.\n"
        << "  --min-frames-between=N Minimum frame gap between selections.\n"
        << "  --min-seconds-between=S Minimum seconds gap between selections.\n"
        << "  --max-cached=N         Max cache size for selected frames.\n"
        << "  --keep-cache=0|1       Keep cached frames after processing (1=keep).\n"
        << "  --enable-cache=0|1     Enable persistent cache for selected frames.\n"
        << "  --no-force-keyframe=0|1 Disable forced keyframe selection.\n"
        << "  --disable-scene-cut=0|1 Disable scene cut selection.\n"
        << "  --min-scene-cut-frames=N  Debounce scene cut by frame count.\n"
        << "  --min-scene-cut-seconds=S Debounce scene cut by seconds.\n"
        << "\n"
        << "Batching:\n"
        << "  --new-batch-mode=0|1   Enable new batch mode (recommended for VLM).\n"
        << "  --legacy-batch-mode=0|1 Disable new batch mode.\n"
        << "  --batch-trigger=N      Trigger batch every N selected frames.\n"
        << "  --max-cache=N          Max cached frames before forced batch.\n"
        << "  --decode-window=N      Decode window size (0 = no fixed window).\n"
        << "  --flush-partial=0|1    Flush partial batch at end.\n"
        << "  --cb_batch_size=N      Continuous batching: requests per CB batch.\n"
        << "  --use_cb=0|1           Enable continuous batching pipeline.\n"
        << "  --cb_multi_thread=0|1  Enable CB engine/statistics threads.\n"
        << "  --new_multithread=0|1  Run CB queue processing in a sub-thread.\n"
        << "\n"
        << "Scheduler (GenAI):\n"
        << "  --max_num_seqs=N       Scheduler max num seqs.\n"
        << "  --dynamic_split_fuse=0|1\n"
        << "\n"
        << "Help:\n"
        << "  --help=1               Print this help message.\n"
        << "  --help | -h | /?       Print this help message.\n";
}

bool ParseCommandLineAndLoadConfig(int argc,
                                   char* argv[],
                                   ParsedArgs& pa,
                                   ParsedArgsW& paw,
                                   DemoConfig& cfg,
                                   std::string& cfgErr,
                                   bool& showHelp)
{
    showHelp = false;
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
        if (HasHelpW(wargc, wargv)) {
            PrintHelp(nullptr);
            showHelp = true;
            LocalFree(wargv);
            return true;
        }
        std::wstring configPathW;
        FindArgValueW(wargc, wargv, L"config", configPathW);
        if (configPathW.empty()) {
            std::filesystem::path p(getExeDirW());
            p /= L"config.json";
            configPathW = p.wstring();
        }
        if (!LoadJSONConfigW(configPathW, cfg, cfgErr)) {
            std::wcerr << L"[Config] Failed to load config: " << configPathW << L" (" << Utf8ToWide(cfgErr) << L")" << std::endl;
            LocalFree(wargv);
            return false;
        }
        ApplyConfig(cfg);
        paw = parseArgsW(wargc, wargv);
        LocalFree(wargv);
        if (paw.ok) {
            pa.ok = true;
            pa.debug = paw.debug;
            pa.use_cb = paw.use_cb;
            pa.cb_batch_size = paw.cb_batch_size;
            pa.input = WideToUtf8(paw.input);
            pa.mode = WideToUtf8(paw.mode);
            pa.prompt = WideToUtf8(paw.prompt);
            pa.outDir = WideToUtf8(paw.outDir);
            pa.configPath = WideToUtf8(paw.configPath);
            pa.videoDir = WideToUtf8(paw.videoDir);
            pa.jsonFile = WideToUtf8(paw.jsonFile);
        } else {
            pa = parseArgs(argc, argv);
        }
        return true;
    }
    // fallback to ANSI
#endif
    if (HasHelpA(argc, argv)) {
        PrintHelp(argc > 0 ? argv[0] : nullptr);
        showHelp = true;
        return true;
    }
    auto getExeDirA = [argv]() -> std::string {
        std::filesystem::path p = argv && argv[0] ? std::filesystem::absolute(argv[0]) : std::filesystem::current_path();
        return p.parent_path().string();
    };
    std::string configPath;
    FindArgValueA(argc, argv, "config", configPath);
    if (configPath.empty()) {
        std::filesystem::path p(getExeDirA());
        p /= "config.json";
        configPath = p.string();
    }
    if (!LoadJSONConfig(configPath, cfg, cfgErr)) {
        std::cerr << "[Config] Failed to load config: " << configPath << " (" << cfgErr << ")" << std::endl;
        return false;
    }
    ApplyConfig(cfg);
    pa = parseArgs(argc, argv);
    return true;
}

bool FinalizeParsedArgs(ParsedArgs& pa,
                        const ParsedArgsW& paw,
                        const DemoConfig& cfg,
                        const char* exeName)
{
    bool hasVideoDir = (!pa.videoDir.empty() || !paw.videoDir.empty());
    if (!pa.ok)
    {
        if (cfg.commonCfg.mode) {
            pa.mode = cfg.commonCfg.mode.value();
        }
        if (cfg.commonCfg.input && pa.input.empty() && !hasVideoDir) {
            pa.input = cfg.commonCfg.input.value();
        }
        if (cfg.commonCfg.out_dir) {
            pa.outDir = cfg.commonCfg.out_dir.value();
        }
        if (cfg.batch.cb_batch_size) pa.cb_batch_size = std::max(0, cfg.batch.cb_batch_size.value());
        pa.ok = (!pa.mode.empty() && (!pa.input.empty() || hasVideoDir));
    }
    if (!pa.ok)
    {
        PrintHelp(exeName);
        return false;
    }
    return true;
}

