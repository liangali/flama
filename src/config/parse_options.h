#pragma once

#include <string>
#include <optional>

struct DemoConfig;

// �����н����ṹ
struct ParsedArgs {
    std::string input;
    std::string mode; // hw | sw
    std::string outDir;
    std::string configPath; // path to JSON config for CB pipeline
    std::string prompt; // VLM prompt override
    std::string videoDir; // directory with multiple videos
    std::string jsonFile; // output JSON file path
    // Scheduler CLI overrides (optional)
    int max_num_seqs = -1; // >0 means set
    bool dynamic_split_fuse = true; // 0=false, 1=true, -1=unset
    bool debug = false;
    bool use_cb = false;
    bool cb_multi_thread = false; // enable CB background threads
    bool new_multithread = false; // run ProcessCBQueueAndReport in a sub thread
    int cb_batch_size = 0;
    bool ok = true;
};

// CLI args use --name=value form only.
// Optional examples:
// --sel-policy=frame|time|mixed
// --frame-interval=N
// --window-seconds=S
// --max-per-window=M
// --min-frames-between=N
// --min-seconds-between=S
// --max-cached=N
// --keep-cache=0|1
// --debug=0|1
ParsedArgs parseArgs(int argc, char* argv[]);

// ���ַ�������������������·����
struct ParsedArgsW {
    std::wstring input;
    std::wstring mode; // hw | sw
    std::wstring outDir;
    std::wstring configPath; // path to JSON config for CB pipeline
    std::wstring prompt; // VLM prompt override
    std::wstring videoDir; // directory with multiple videos
    std::wstring jsonFile; // output JSON file path
    // Scheduler CLI overrides (optional)
    int max_num_seqs = -1; // >0 means set
    bool dynamic_split_fuse = true; // 0=false, 1=true, -1=unset
    bool debug = false;
    bool use_cb = false;
    bool cb_multi_thread = false; // enable CB background threads
    bool new_multithread = false; // run ProcessCBQueueAndReport in a sub thread
    int cb_batch_size = 0;
    bool ok = true;
};

ParsedArgsW parseArgsW(int argc, wchar_t* argv[]);

// Print full help message (all CLI args use --name=value)
void PrintHelp(const char* exeName);

// Parse CLI, load JSON config, and apply config to globals.
// Returns true on success or when --help is requested.
bool ParseCommandLineAndLoadConfig(int argc,
                                   char* argv[],
                                   ParsedArgs& pa,
                                   ParsedArgsW& paw,
                                   DemoConfig& cfg,
                                   std::string& cfgErr,
                                   bool& showHelp);

// Apply config defaults for missing CLI values and validate required args.
// Returns false if required args are still missing (also prints help).
bool FinalizeParsedArgs(ParsedArgs& pa,
                        const ParsedArgsW& paw,
                        const DemoConfig& cfg,
                        const char* exeName);
