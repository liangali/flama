#include <fstream>
#include <sstream>
#include <algorithm>

#include "parse_options.h"
#include "../core/util.h"
// �����н���ʵ��
ParsedArgs parseArgs(int argc, char* argv[]) {
    ParsedArgs pa;
    if (argc < 3) { pa.ok = false; return pa; }
    pa.input = argv[1];
    pa.mode = argv[2];
    pa.configPath = "config/config.json"; // default
    // outDir ������������������Ŀ¼����Բ�����
    if (argc >= 4 && argv[3] && argv[3][0] != '-') {
        pa.outDir = argv[3];
    }
    else {
        pa.outDir = ".";
    }

    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) continue;
        std::string a = argv[i];
        if (a == "--config" && (i + 1) < argc && argv[i + 1] && argv[i + 1][0] != '-') {
            pa.configPath = argv[++i];
            continue;
        }
        else if (a.rfind("--config=", 0) == 0) {
            pa.configPath = a.substr(9);
            continue;
        }
        if (a == "--debug" || a == "-d" || a == "debug" || a == "DEBUG") {
            pa.debug = true;
            g_commonConfig.debug = true;
        }
        else if (a.rfind("--max_num_seqs=", 0) == 0) {
            int v = std::stoi(a.substr(16).c_str());
            if (v > 0) {
                pa.max_num_seqs = v; g_vlmConfig.shedulerConfig.max_num_seqs = v;
            }
        }
        else if (a.rfind("--dynamic_split_fuse=", 0) == 0) {
            std::string v = a.substr(21);
            if (v == "0" || v == "false" || v == "False" || v == "FALSE")
            {
                pa.dynamic_split_fuse = false;
                g_vlmConfig.shedulerConfig.dynamic_split_fuse = false;
            }
            else if (v == "1" || v == "true" || v == "True" || v == "TRUE")
            {
                pa.dynamic_split_fuse = true;
                g_vlmConfig.shedulerConfig.dynamic_split_fuse = true;
            }
        }
        else if (a == "--use_cb") {
            pa.use_cb = true;
            g_commonConfig.use_cb = true;
            g_vlmConfig.enable_continuous_batching = true;
        }
                else if (a == "--new_multithread" || a == "--new-multithread") {
                    pa.new_multithread = true;
                    g_commonConfig.new_multithread = true;
                }
                else if (a.rfind("--new_multithread=", 0) == 0 || a.rfind("--new-multithread=", 0) == 0) {
                    std::string v = a.substr(a.find('=') + 1);
                    bool flag = (v == "1" || v == "true" || v == "True" || v == "TRUE");
                    pa.new_multithread = flag;
                    g_commonConfig.new_multithread = flag;
                }
        else if (a == "--cb_multi_thread" || a == "--cb-multi-thread") {
            pa.cb_multi_thread = true;
            g_commonConfig.cb_multi_thread = true;
        }
        else if (a.rfind("--cb_multi_thread=", 0) == 0 || a.rfind("--cb-multi-thread=", 0) == 0) {
            std::string v = a.substr(a.find('=') + 1);
            bool flag = (v == "1" || v == "true" || v == "True" || v == "TRUE");
            pa.cb_multi_thread = flag;
            g_commonConfig.cb_multi_thread = flag;
        }
        else if (a.rfind("--cb_batch_size=", 0) == 0) {
            pa.cb_batch_size = std::max(0, std::stoi(a.substr(16).c_str()));
            g_batchConfig.cb_batch_size = std::max(0, std::stoi(a.substr(16).c_str()));
        }
        else if (a.rfind("--sel-policy=", 0) == 0) {
            std::string v = a.substr(13);
            if (v == "frame") g_fsConfig.policy = FSPolicy::FrameInterval;
            else if (v == "time") g_fsConfig.policy = FSPolicy::TimeWindowQuota;
            else if (v == "key") g_fsConfig.policy = FSPolicy::KeyframePriority;
            else if (v == "mixed-key") g_fsConfig.policy = FSPolicy::MixedKeyframe;
            else g_fsConfig.policy = FSPolicy::Mixed;
        }
        else if (a.rfind("--frame-interval=", 0) == 0) {
            g_fsConfig.frame_interval = std::max(1, std::stoi(a.substr(17)));
        }
        else if (a.rfind("--window-seconds=", 0) == 0) {
            g_fsConfig.window_seconds = std::stod(a.substr(17));
            if (g_fsConfig.window_seconds < 0.0) g_fsConfig.window_seconds = 0.0;
        }
        else if (a.rfind("--max-per-window=", 0) == 0) {
            g_fsConfig.max_per_window = std::stoi(a.substr(17));
            if (g_fsConfig.max_per_window < 0) g_fsConfig.max_per_window = 0;
        }
        else if (a.rfind("--min-frames-between=", 0) == 0) {
            g_fsConfig.min_frames_between = std::stoi(a.substr(21));
            if (g_fsConfig.min_frames_between < 0) g_fsConfig.min_frames_between = 0;
        }
        else if (a.rfind("--min-seconds-between=", 0) == 0) {
            g_fsConfig.min_seconds_between = std::stod(a.substr(22));
            if (g_fsConfig.min_seconds_between < 0.0) g_fsConfig.min_seconds_between = 0.0;
        }
        else if (a.rfind("--max-cached=", 0) == 0) {
            //g_fsConfig.max_cached = std::stoi(a.substr(13));
            //if (g_fsConfig.max_cached < 1) g_fsConfig.max_cached = 1;
        }
        else if (a == "--keep-cache") {
            g_fsConfig.remove_after_process = false;
        }
        else if (a == "--no-force-keyframe") {
            g_fsConfig.force_keyframe = false;
        }
        else if (a == "--disable-scene-cut") {
            g_fsConfig.enable_scene_cut = false;
        }
        else if (a.rfind("--min-scene-cut-frames=", 0) == 0) {
            g_fsConfig.min_frames_between_scene_cut = std::stoi(a.substr(23));
            if (g_fsConfig.min_frames_between_scene_cut < 0) g_fsConfig.min_frames_between_scene_cut = 0;
        }
        else if (a.rfind("--min-scene-cut-seconds=", 0) == 0) {
            g_fsConfig.min_seconds_between_scene_cut = std::stod(a.substr(24));
            if (g_fsConfig.min_seconds_between_scene_cut < 0.0) g_fsConfig.min_seconds_between_scene_cut = 0.0;
        }
        else if (a == "--enable-cache" || a == "--cache") {
            g_fsConfig.enable_cache = true;
        }
        else if (a.rfind("--batch-trigger=", 0) == 0) {
            g_batchConfig.batch_trigger = std::max(1, std::stoi(a.substr(16)));
        }
        else if (a.rfind("--max-cache=", 0) == 0) {
            g_batchConfig.max_cache = std::max(1, std::stoi(a.substr(12)));
        }
        else if (a.rfind("--decode-window=", 0) == 0) {
            g_batchConfig.decode_window = std::max(0, std::stoi(a.substr(15)));
        }
        else if (a == "--flush-partial=0") {
            g_batchConfig.flush_partial = false;
        }
        else if (a == "--flush-partial=1") {
            g_batchConfig.flush_partial = true;
        }
        else if (a == "--new-batch-mode") {
            g_batchConfig.new_batch_mode = true;
        }
        else if (a == "--legacy-batch-mode") {
            g_batchConfig.new_batch_mode = false;
        }
    }
    return pa;
}

// ���ַ�����ʵ��
ParsedArgsW parseArgsW(int argc, wchar_t* argv[]) {
    ParsedArgsW pa;
    if (argc < 3) { pa.ok = false; return pa; }
    pa.input = argv[1];
    pa.mode = argv[2];
    pa.configPath = L"config/config.json"; // default
    if (argc >= 4 && argv[3] && argv[3][0] != L'-') {
        pa.outDir = argv[3];
    }
    else {
        pa.outDir = L".";
    }

    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) continue;
        std::wstring a = argv[i];
        if (a == L"--config" && (i + 1) < argc && argv[i + 1] && argv[i + 1][0] != L'-') {
            pa.configPath = argv[++i];
            continue;
        }
        else if (a.rfind(L"--config=", 0) == 0) {
            pa.configPath = a.substr(9);
            continue;
        }
        if (a == L"--debug" || a == L"-d" || a == L"debug" || a == L"DEBUG") {
            pa.debug = true;
            g_commonConfig.debug = true;
        }
        else if (a.rfind(L"--max_num_seqs=", 0) == 0) {
            int v = _wtoi(a.substr(16).c_str());
            if (v > 0) {
                pa.max_num_seqs = v; g_vlmConfig.shedulerConfig.max_num_seqs = v;
            }
        }
        else if (a.rfind(L"--dynamic_split_fuse=", 0) == 0) {
            std::wstring v = a.substr(21);
            if (v == L"0" || v == L"false" || v == L"False" || v == L"FALSE")
            {
                pa.dynamic_split_fuse = false;
                g_vlmConfig.shedulerConfig.dynamic_split_fuse = false;
            }
            else if (v == L"1" || v == L"true" || v == L"True" || v == L"TRUE")
            {
                pa.dynamic_split_fuse = true;
                g_vlmConfig.shedulerConfig.dynamic_split_fuse = true;
            }
        }
        else if (a == L"--use_cb") {
            pa.use_cb = true;
            g_commonConfig.use_cb = true;
            g_vlmConfig.enable_continuous_batching = true;
        }
                else if (a == L"--new_multithread" || a == L"--new-multithread") {
                    pa.new_multithread = true;
                    g_commonConfig.new_multithread = true;
                }
                else if (a.rfind(L"--new_multithread=", 0) == 0 || a.rfind(L"--new-multithread=", 0) == 0) {
                    std::wstring v = a.substr(a.find(L'=') + 1);
                    bool flag = (v == L"1" || v == L"true" || v == L"True" || v == L"TRUE");
                    pa.new_multithread = flag;
                    g_commonConfig.new_multithread = flag;
                }
        else if (a == L"--cb_multi_thread" || a == L"--cb-multi-thread") {
            pa.cb_multi_thread = true;
            g_commonConfig.cb_multi_thread = true;
        }
        else if (a.rfind(L"--cb_multi_thread=", 0) == 0 || a.rfind(L"--cb-multi-thread=", 0) == 0) {
            std::wstring v = a.substr(a.find(L'=') + 1);
            bool flag = (v == L"1" || v == L"true" || v == L"True" || v == L"TRUE");
            pa.cb_multi_thread = flag;
            g_commonConfig.cb_multi_thread = flag;
        }
        else if (a.rfind(L"--cb_batch_size=", 0) == 0) {
            pa.cb_batch_size = std::max(0, _wtoi(a.substr(16).c_str()));
            g_batchConfig.cb_batch_size = std::max(0, _wtoi(a.substr(16).c_str()));
        }
        else if (a.rfind(L"--sel-policy=", 0) == 0) {
            std::wstring v = a.substr(13);
            if (v == L"frame") g_fsConfig.policy = FSPolicy::FrameInterval;
            else if (v == L"time") g_fsConfig.policy = FSPolicy::TimeWindowQuota;
            else if (v == L"key") g_fsConfig.policy = FSPolicy::KeyframePriority;
            else if (v == L"mixed-key") g_fsConfig.policy = FSPolicy::MixedKeyframe;
            else g_fsConfig.policy = FSPolicy::Mixed;
        }
        else if (a.rfind(L"--frame-interval=", 0) == 0) {
            g_fsConfig.frame_interval = std::max(1, _wtoi(a.substr(17).c_str()));
        }
        else if (a.rfind(L"--window-seconds=", 0) == 0) {
            g_fsConfig.window_seconds = _wtof(a.substr(17).c_str());
            if (g_fsConfig.window_seconds < 0.0) g_fsConfig.window_seconds = 0.0;
        }
        else if (a.rfind(L"--max-per-window=", 0) == 0) {
            g_fsConfig.max_per_window = _wtoi(a.substr(17).c_str());
            if (g_fsConfig.max_per_window < 0) g_fsConfig.max_per_window = 0;
        }
        else if (a.rfind(L"--min-frames-between=", 0) == 0) {
            g_fsConfig.min_frames_between = _wtoi(a.substr(21).c_str());
            if (g_fsConfig.min_frames_between < 0) g_fsConfig.min_frames_between = 0;
        }
        else if (a.rfind(L"--min-seconds-between=", 0) == 0) {
            g_fsConfig.min_seconds_between = _wtof(a.substr(22).c_str());
            if (g_fsConfig.min_seconds_between < 0.0) g_fsConfig.min_seconds_between = 0.0;
        }
        else if (a.rfind(L"--max-cached=", 0) == 0) {
            //g_fsConfig.max_cached = _wtoi(a.substr(13).c_str());
            //if (g_fsConfig.max_cached < 1) g_fsConfig.max_cached = 1;
        }
        else if (a == L"--keep-cache") {
            g_fsConfig.remove_after_process = false;
        }
        else if (a == L"--no-force-keyframe") {
            g_fsConfig.force_keyframe = false;
        }
        else if (a == L"--disable-scene-cut") {
            g_fsConfig.enable_scene_cut = false;
        }
        else if (a.rfind(L"--min-scene-cut-frames=", 0) == 0) {
            g_fsConfig.min_frames_between_scene_cut = _wtoi(a.substr(23).c_str());
            if (g_fsConfig.min_frames_between_scene_cut < 0) g_fsConfig.min_frames_between_scene_cut = 0;
        }
        else if (a.rfind(L"--min-scene-cut-seconds=", 0) == 0) {
            g_fsConfig.min_seconds_between_scene_cut = _wtof(a.substr(24).c_str());
            if (g_fsConfig.min_seconds_between_scene_cut < 0.0) g_fsConfig.min_seconds_between_scene_cut = 0.0;
        }
        else if (a == L"--enable-cache" || a == L"--cache") {
            g_fsConfig.enable_cache = true;
        }
        else if (a.rfind(L"--batch-trigger=", 0) == 0) {
            g_batchConfig.batch_trigger = std::max(1, _wtoi(a.substr(16).c_str()));
        }
        else if (a.rfind(L"--max-cache=", 0) == 0) {
            g_batchConfig.max_cache = std::max(1, _wtoi(a.substr(12).c_str()));
        }
        else if (a.rfind(L"--decode-window=", 0) == 0) {
            g_batchConfig.decode_window = std::max(0, _wtoi(a.substr(15).c_str()));
        }
        else if (a == L"--flush-partial=0") {
            g_batchConfig.flush_partial = false;
        }
        else if (a == L"--flush-partial=1") {
            g_batchConfig.flush_partial = true;
        }
        else if (a == L"--new-batch-mode") {
            g_batchConfig.new_batch_mode = true;
        }
        else if (a == L"--legacy-batch-mode") {
            g_batchConfig.new_batch_mode = false;
        }
    }
    return pa;
}

