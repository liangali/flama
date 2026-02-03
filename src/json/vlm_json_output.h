//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#pragma once

#include <string>
#include <vector>
#include <filesystem>

struct VlmJsonSegment
{
    int seg_id = 0;
    double seg_start = 0.0;
    double seg_end = 0.0;
    double seg_dur = 0.0;
    std::string seg_desc;
};

struct VlmJsonVideo
{
    std::string input_video;
    std::string prompt;
    std::vector<VlmJsonSegment> segments;
    int next_seg_id = 0;
};

class VlmJsonCollector
{
public:
    void StartVideo(const std::string &input_video, const std::string &prompt);
    void AddSegment(double seg_start, double seg_end, const std::string &desc);
    std::vector<VlmJsonVideo> Snapshot() const;
    void Reset();

private:
    std::vector<VlmJsonVideo> videos_;
    size_t current_index_ = 0;
};

// Global VLM JSON collector instance
extern VlmJsonCollector g_vlmJsonCollector;

// Write VLM results to JSON file
bool WriteVlmJsonFile(const std::filesystem::path &outPath, const std::vector<VlmJsonVideo> &videos);
