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
        else if (key == "prompt") {
            pa.prompt = val;
            if (!pa.prompt.empty()) g_commonConfig.prompt_video = pa.prompt;
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
        else if (key == L"prompt") {
            pa.prompt = val;
            if (!pa.prompt.empty()) g_commonConfig.prompt_video = WideToUtf8(pa.prompt);
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
        << "  --input=PATH           Input video file path.\n"
        << "  --video_dir=PATH       Directory of video files (sorted by filename).\n"
        << "  --mode=hw|sw           Decode path: hw (D3D11) or sw.\n"
        << "  --out_dir=PATH         Output directory for CSV logs.\n"
        << "  --config=PATH          JSON config path (default: exe_dir/config.json).\n"
        << "  --json_file=PATH       Output JSON file for VLM results (default: ./output_vlm.json).\n"
        << "  --prompt=TEXT          Override VLM prompt for video batches.\n"
        << "  --debug=0|1            Enable debug logging.\n"

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
        pa.ok = (!pa.mode.empty() && (!pa.input.empty() || hasVideoDir));
    }
    if (!pa.ok)
    {
        PrintHelp(exeName);
        return false;
    }
    return true;
}

