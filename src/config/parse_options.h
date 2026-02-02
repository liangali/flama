#pragma once

#include <string>
#include <optional>

struct DemoConfig;
struct ParsedArgs {
    std::string input;
    std::string mode; // hw | sw
    std::string configPath; // path to JSON config for CB pipeline
    std::string prompt; // VLM prompt override
    std::string videoDir; // directory with multiple videos
    std::string jsonFile; // output JSON file path
    bool debug = false;
    bool ok = true;
};

ParsedArgs parseArgs(int argc, char* argv[]);

struct ParsedArgsW {
    std::wstring input;
    std::wstring mode; // hw | sw
    std::wstring configPath; // path to JSON config for CB pipeline
    std::wstring prompt; // VLM prompt override
    std::wstring videoDir; // directory with multiple videos
    std::wstring jsonFile; // output JSON file path
    bool debug = false;
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
