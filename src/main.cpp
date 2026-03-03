//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

// Standard library
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cstdlib>

// Windows headers
#ifdef _WIN32
#include <windows.h>
#endif

// FFmpeg headers
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

// New modular includes
#include "utils/path_utils.h"
#include "utils/util.h"
#include "utils/debug.h"
#include "utils/profiling.h"
#include "config/config_logging.h"
#include "config/json_config.hpp"
#include "config/parse_options.h"
#include "json/vlm_json_output.h"
#include "video/ffmpeg_decoder.h"
#include "video/decode_pipeline.h"
#include "batch/batch_state.h"
#include "inference/cb_processing.h"
#include "inference/vlm_chat.h"

int main(int argc, char *argv[])
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // Load JSON config early (resolve --config from args)
    DemoConfig cfg;
    std::string cfgErr;
    ParsedArgs pa{};
    ParsedArgsW paw{};
    bool showHelp = false;
    if (!ParseCommandLineAndLoadConfig(argc, argv, pa, paw, cfg, cfgErr, showHelp))
        return 1;
    if (showHelp)
        return 0;
    if (!FinalizeParsedArgs(pa, paw, cfg, argv && argv[0] ? argv[0] : nullptr))
        return 1;

    bool csvEnabled = pa.debug;
    prof::FrameProfiler::SetCsvEnabled(csvEnabled);
    if (pa.debug || std::getenv("VLM_DEBUG"))
    {
        g_commonConfig.debug = true;
        SetDebugMode(g_commonConfig.debug);
    }

    // Apply CLI overrides into globals immediately after parsing
    if (!pa.input.empty()) g_commonConfig.input_video_path = pa.input;
    if (!pa.mode.empty())  g_commonConfig.hw_decode = (pa.mode == "hw" || pa.mode == "HW");

    DBG_LOG(std::string("[Main] DebugMode=") + (IsDebugMode() ? "ON" : "OFF"));

    // Dump loaded config sections for visibility
    std::cout << "Log DemoConfig start" << std::endl;
    LogDemoConfig(cfg);
    std::cout << "Log DemoConfig end" << std::endl;

    // Log resolved global runtime configs
    std::cout << "Log g_fsConfig/g_batchConfig/g_commonConfig/g_vlmConfig start" << std::endl;
    LogBatchConfig(g_batchConfig);
    LogFSConfig(g_fsConfig);
    LogBatchConfig(g_batchConfig);
    LogCommonConfig(g_commonConfig);
    LogVLMConfig(g_vlmConfig);
    std::cout << "Log g_fsConfig/g_batchConfig/g_commonConfig/g_vlmConfig end" << std::endl;

    g_vlmJsonCollector.Reset();
    ResetVLMTokenTotals();

    bool useHardware = g_commonConfig.hw_decode;
    bool useSoftware = !useHardware;

    // Resolve use_cb and cb_batch_size from globals (after CLI overrides)
    use_cb = g_vlmConfig.enable_continuous_batching && g_commonConfig.use_cb;
    cb_batch_size = g_batchConfig.cb_batch_size;
    DBG_LOGF(" use_cb=%d cb_batch_size=%d, useHardware = %d, useSoftware = %d", use_cb ? 1 : 0, cb_batch_size, useHardware, useSoftware);

    const char *hw_device_type = "d3d11va";

    init_ffmpeg();

    // Resolve JSON output path
    std::filesystem::path jsonOutPath;
#ifdef _WIN32
    if (!paw.jsonFile.empty())
        jsonOutPath = std::filesystem::path(paw.jsonFile);
    else if (!pa.jsonFile.empty())
        jsonOutPath = std::filesystem::path(Utf8ToWide(pa.jsonFile));
    else
        jsonOutPath = std::filesystem::current_path() / "output_vlm.json";
#else
    if (!pa.jsonFile.empty())
        jsonOutPath = std::filesystem::path(pa.jsonFile);
    else
        jsonOutPath = std::filesystem::current_path() / "output_vlm.json";
#endif

    // Collect input videos
    std::vector<std::filesystem::path> inputVideos;
#ifdef _WIN32
    if (!paw.videoDir.empty() || !pa.videoDir.empty())
    {
        std::filesystem::path videoDirPath = !paw.videoDir.empty() ? std::filesystem::path(paw.videoDir)
                                                                   : std::filesystem::path(Utf8ToWide(pa.videoDir));
        inputVideos = CollectVideoFiles(videoDirPath);
        if (inputVideos.empty())
        {
            std::cerr << "[Input] No video files found in: " << WideToUtf8(videoDirPath.wstring()) << std::endl;
            return 1;
        }
    }
    else
    {
        std::wstring singleInputW;
        if (!paw.input.empty())
            singleInputW = paw.input;
        else if (!pa.input.empty())
            singleInputW = Utf8ToWide(pa.input);
        else
            singleInputW = Utf8ToWide(g_commonConfig.input_video_path);
        std::filesystem::path singlePath(singleInputW);
        inputVideos.push_back(singlePath);
    }
#else
    if (!pa.videoDir.empty())
    {
        inputVideos = CollectVideoFiles(std::filesystem::path(pa.videoDir));
        if (inputVideos.empty())
        {
            std::cerr << "[Input] No video files found in: " << pa.videoDir << std::endl;
            return 1;
        }
    }
    else
    {
        std::filesystem::path singlePath = !pa.input.empty() ? std::filesystem::path(pa.input)
                                                             : std::filesystem::path(g_commonConfig.input_video_path);
        inputVideos.push_back(singlePath);
    }
#endif

    bool all_ok = true;
    for (const auto &rawPath : inputVideos)
    {
        std::filesystem::path inputPath = std::filesystem::absolute(rawPath);
#ifdef _WIN32
        std::wstring inputPathW = inputPath.wstring();
        std::string input_file = WideToUtf8(inputPathW);
#else
        std::string input_file = inputPath.string();
#endif
        g_commonConfig.input_video_path = input_file;
        SetVLMInputFile(input_file);

        // Reset state for each video
        g_batchState.reset();
        g_batchState.batchIndex = 0;
        g_cbInferenceQueue.clear();
        g_cb_pending_batches = 0;
        g_cb_enqueued_batches = 0;
        cb_timer = true;
        report_idx = 0;
        g_cb_stop.store(false);
        g_sw_inference_total_us = 0;
        g_sw_pipeline_total_us = 0;
        g_hw_inference_total_us = 0;
        g_hw_pipeline_total_us = 0;
        ResetSampleState();

        // Open video file
        AVFormatContext *format_context = nullptr;
        if (avformat_open_input(&format_context, input_file.c_str(), nullptr, nullptr) < 0)
        {
            std::cerr << "Failed to open input file: " << input_file << std::endl;
            all_ok = false;
            continue;
        }

        std::cout << "Opened input video file " << input_file << " file format: " << format_context->iformat->name << std::endl;

        if (avformat_find_stream_info(format_context, nullptr) < 0)
        {
            std::cerr << "Failed to find stream information." << std::endl;
            avformat_close_input(&format_context);
            all_ok = false;
            continue;
        }

        int stream_index = find_video_stream(format_context);
        if (stream_index < 0)
        {
            std::cerr << "No video stream found." << std::endl;
            avformat_close_input(&format_context);
            all_ok = false;
            continue;
        }

        g_vlmJsonCollector.StartVideo(input_file, g_commonConfig.prompt_video);

        // Derive width/height and simple codec string
        int srcW = 0, srcH = 0;
        std::wstring codecSimple = L"";
        if (format_context && stream_index >= 0)
        {
            AVCodecParameters *codec_params = format_context->streams[stream_index]->codecpar;
            srcW = codec_params->width;
            srcH = codec_params->height;
            switch (codec_params->codec_id)
            {
            case AV_CODEC_ID_H264:
                codecSimple = L"avc";
                break;
            case AV_CODEC_ID_HEVC:
                codecSimple = L"hevc";
                break;
            default:
                codecSimple = L"unknown";
                break;
            }
        }

        // Setup profiling output files
        std::string clipName = inputPath.stem().string();
#ifdef _WIN32
        std::filesystem::path outBaseW = std::filesystem::current_path();
        std::wstring filename_w = ExtractFilenameWithoutExt(inputPathW);
        std::filesystem::path profPathW = outBaseW / (filename_w + L"_profile_" + (useSoftware ? L"sw" : L"hw") + L".csv");
        std::filesystem::path resultPathW = outBaseW / (filename_w + L"_results_" + (useSoftware ? L"sw" : L"hw") + L".csv");
        std::filesystem::path batchPathW = outBaseW / (filename_w + L"_batch_" + (useSoftware ? L"sw" : L"hw") + L".csv");

        if (srcW > 0 && srcH > 0)
        {
            std::wstring suffix = L"_" + codecSimple + L"_" + std::to_wstring(srcW) + L"x" + std::to_wstring(srcH);
            profPathW = outBaseW / ((filename_w + suffix + L"_profile_" + (useSoftware ? L"sw" : L"hw")) + L".csv");
            resultPathW = outBaseW / ((filename_w + suffix + L"_results_" + (useSoftware ? L"sw" : L"hw")) + L".csv");
            batchPathW = outBaseW / ((filename_w + suffix + L"_batch_" + (useSoftware ? L"sw" : L"hw")) + L".csv");
        }

        if (csvEnabled)
        {
            SetVLMResultFileW(resultPathW.wstring());
            prof::FrameProfiler::SetOutputFileW(profPathW.wstring());
            DBG_LOG(std::string("[PROF] Output -> ") + WideToUtf8(profPathW.wstring()));
            prof::BatchAggregator::Get().SetOutputFileW(batchPathW.wstring());
            DBG_LOG(std::string("[BATCH] Output -> ") + WideToUtf8(batchPathW.wstring()));
        }
#else
        std::filesystem::path outBase = std::filesystem::current_path();
        std::string profFile = (outBase / (clipName + std::string("_profile_") + (useSoftware ? "sw" : "hw") + ".csv")).string();
        std::string resultFile = (outBase / (clipName + std::string("_results_") + (useSoftware ? "sw" : "hw") + ".csv")).string();
        std::string batchFile = (outBase / (clipName + std::string("_batch_") + (useSoftware ? "sw" : "hw") + ".csv")).string();

        if (srcW > 0 && srcH > 0)
        {
            std::string suffix = std::string("_") + WideToUtf8(codecSimple) + "_" + std::to_string(srcW) + "x" + std::to_string(srcH);
            profFile = (outBase / (clipName + suffix + std::string("_profile_") + (useSoftware ? "sw" : "hw") + ".csv")).string();
            resultFile = (outBase / (clipName + suffix + std::string("_results_") + (useSoftware ? "sw" : "hw") + ".csv")).string();
            batchFile = (outBase / (clipName + suffix + std::string("_batch_") + (useSoftware ? "sw" : "hw") + ".csv")).string();
        }

        if (csvEnabled)
        {
            SetVLMResultFile(resultFile);
            prof::FrameProfiler::SetOutputFile(profFile);
            DBG_LOG(std::string("[PROF] Output -> ") + profFile);
            prof::BatchAggregator::Get().SetOutputFile(batchFile);
            DBG_LOG(std::string("[BATCH] Output -> ") + batchFile);
        }
#endif

        // Open decoder and process video
        AVCodecContext *codec_context = nullptr;
        if (useSoftware)
        {
            DBG_LOG("[Main] Using software decoding path.");
            codec_context = open_decoder_sw(format_context, stream_index);
            if (!codec_context)
            {
                avformat_close_input(&format_context);
                all_ok = false;
                continue;
            }
            auto decode_start = std::chrono::high_resolution_clock::now();
            decode_frames_sw(format_context, codec_context, stream_index);
            auto decode_end = std::chrono::high_resolution_clock::now();

            std::cout << "total time of entire pipeline: " << std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count() << " ms" << std::endl;
            std::cout << "[SW] total inference time: " << (g_sw_inference_total_us / 1000) << " ms" << std::endl;
            std::cout << "[SW] total decode_frames_sw time: " << (g_sw_pipeline_total_us / 1000) << " ms" << std::endl;
        }
        else
        {
            DBG_LOG(std::string("[Main] Using hardware decoding path: ") + hw_device_type);
            codec_context = open_decoder(format_context, stream_index, hw_device_type);
            if (!codec_context)
            {
                avformat_close_input(&format_context);
                all_ok = false;
                continue;
            }
            auto decode_start = std::chrono::high_resolution_clock::now();
            decode_frames(format_context, codec_context, stream_index);
            auto decode_end = std::chrono::high_resolution_clock::now();

            std::cout << "[HW] total inference time: " << (g_hw_inference_total_us / 1000) << " ms" << std::endl;
            std::cout << "[HW] total decode_frames time: " << (g_hw_pipeline_total_us / 1000) << " ms" << std::endl;
        }

        cleanup(format_context, codec_context);
        if (useHardware)
        {
            ResetHwResources();
        }

        DBG_LOG("[VLM] Result file finalized.");
        prof::Summary::Get().PrintSummary();
    }

    // Write JSON output
    try
    {
        if (jsonOutPath.has_parent_path())
            std::filesystem::create_directories(jsonOutPath.parent_path());
    }
    catch (const std::exception &e)
    {
        std::cerr << "[JSON] Failed to create output directory: " << e.what() << std::endl;
    }
    if (!WriteVlmJsonFile(jsonOutPath, g_vlmJsonCollector.Snapshot()))
        all_ok = false;

    PrintVLMTokenTotals("[VLM] Run totals");

    return all_ok ? 0 : 1;
}
