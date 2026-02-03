//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "vlm_json_output.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

// Global VLM JSON collector instance
VlmJsonCollector g_vlmJsonCollector;

void VlmJsonCollector::StartVideo(const std::string &input_video, const std::string &prompt)
{
    VlmJsonVideo v;
    v.input_video = input_video;
    v.prompt = prompt;
    videos_.push_back(std::move(v));
    current_index_ = videos_.empty() ? 0 : (videos_.size() - 1);
}

void VlmJsonCollector::AddSegment(double seg_start, double seg_end, const std::string &desc)
{
    if (videos_.empty())
        return;
    VlmJsonVideo &v = videos_[current_index_];
    VlmJsonSegment s;
    s.seg_id = v.next_seg_id++;
    s.seg_start = seg_start;
    s.seg_end = seg_end;
    s.seg_dur = (seg_end >= seg_start) ? (seg_end - seg_start) : 0.0;
    s.seg_desc = desc;
    v.segments.push_back(std::move(s));
}

std::vector<VlmJsonVideo> VlmJsonCollector::Snapshot() const
{
    return videos_;
}

void VlmJsonCollector::Reset()
{
    videos_.clear();
    current_index_ = 0;
}

bool WriteVlmJsonFile(const std::filesystem::path &outPath, const std::vector<VlmJsonVideo> &videos)
{
    auto FormatFixed3 = [](double v) {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss << std::setprecision(3) << v;
        return oss.str();
    };
    nlohmann::ordered_json j;
    j["processed_videos"] = nlohmann::json::array();
    for (const auto &video : videos)
    {
        nlohmann::ordered_json v;
        v["input_video"] = video.input_video;
        v["prompt"] = video.prompt;
        v["segments"] = nlohmann::json::array();
        for (const auto &seg : video.segments)
        {
            nlohmann::ordered_json s;
            s["seg_id"] = seg.seg_id;
            s["seg_start"] = FormatFixed3(seg.seg_start);
            s["seg_end"] = FormatFixed3(seg.seg_end);
            s["seg_dur"] = FormatFixed3(seg.seg_dur);
            s["seg_desc"] = seg.seg_desc;
            v["segments"].push_back(std::move(s));
        }
        j["processed_videos"].push_back(std::move(v));
    }
    std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "[JSON] Failed to open output file: " << outPath.string() << std::endl;
        return false;
    }
    out << j.dump(2);
    return true;
}
