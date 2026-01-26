#include "FrameSelector.h"
#include <algorithm>
#include <iostream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

void FrameSelector::Reset() {
    total_decoded_ = 0;
    last_selected_frame_index_ = -INT_MAX;
    last_selected_pts_sec_ = -DBL_MAX;
    window_start_sec_ = -DBL_MAX;
    window_selected_count_ = 0;
    ClearCache();
}

double FrameSelector::ExtractPtsSec(AVFrame* f, AVRational tb) {
    if (!f) return 0.0;
    int64_t pts = (f->pts == AV_NOPTS_VALUE ? f->best_effort_timestamp : f->pts);
    if (pts == AV_NOPTS_VALUE) pts = 0;
    double base = av_q2d(tb);
    return base * static_cast<double>(pts);
}

bool FrameSelector::DecidePick(double pts_sec) {
    // 帧间隔策略
    bool byFrame = (cfg_.frame_interval <= 1) ? true : (total_decoded_ % cfg_.frame_interval == 0);

    // 时间窗口策略
    bool byTime = false;
    if (cfg_.window_seconds > 0.0 && cfg_.max_per_window > 0) {
        if (!std::isfinite(window_start_sec_)) {
            window_start_sec_ = pts_sec;
            window_selected_count_ = 0;
        }
        if (pts_sec - window_start_sec_ >= cfg_.window_seconds) {
            window_start_sec_ = pts_sec;
            window_selected_count_ = 0;
        }
        if (window_selected_count_ < cfg_.max_per_window) {
            byTime = true;
            window_selected_count_++;
        }
    }

    // 关键帧判断：AVFrame 的 key_frame 标志（部分编解码器可能未填充）
    bool isKeyframe = false; // 在 AcceptDecodedFrame 中重新获取 frame->key_frame
    // 暂时在这里不获取 frame 指针，仅通过 AcceptDecodedFrame 传入时再判断

    bool pick = false;
    switch (cfg_.policy) {
        case FSPolicy::FrameInterval: pick = byFrame; break;
        case FSPolicy::TimeWindowQuota: pick = byTime; break;
        case FSPolicy::Mixed: pick = (byFrame || byTime); break;
        case FSPolicy::KeyframePriority: // 关键帧优先，其余按帧间隔补充
            // 在 AcceptDecodedFrame 中若是关键帧将强制选中；此处仅保留非关键帧逻辑
            pick = byFrame || byTime; break;
        case FSPolicy::MixedKeyframe: // 关键帧必选 或 满足其它任一
            pick = byFrame || byTime; break;
    }

    // 最小间隔约束
    if (pick && cfg_.min_frames_between > 0 &&
        (total_decoded_ - last_selected_frame_index_) < cfg_.min_frames_between) pick = false;
    if (pick && cfg_.min_seconds_between > 0.0 && std::isfinite(last_selected_pts_sec_) &&
        (pts_sec - last_selected_pts_sec_) < cfg_.min_seconds_between) pick = false;
    return pick;
}

bool FrameSelector::AcceptDecodedFrame(AVFrame* frame, AVRational time_base, AVFrame*& outSelected) {
    outSelected = nullptr;
    total_decoded_++;
    double pts_sec = ExtractPtsSec(frame, time_base);
    bool basePick = DecidePick(pts_sec);
    bool isKeyframe = (frame && ((frame->flags & AV_FRAME_FLAG_KEY) != 0));
    bool sceneCutPick = false;
    if (pending_scene_cut_) {
        // 场景切换触发后检查防抖约束
        if (SceneCutAllowed(pts_sec)) {
            sceneCutPick = true;
            last_scene_cut_frame_index_ = total_decoded_;
            last_scene_cut_pts_sec_ = pts_sec;
        }
        pending_scene_cut_ = false; // 消耗一次触发
    }

    bool pick = basePick;
    if (cfg_.policy == FSPolicy::KeyframePriority || cfg_.policy == FSPolicy::MixedKeyframe) {
        if (isKeyframe && cfg_.force_keyframe) pick = true; // 关键帧强制选中
    }
    if (cfg_.enable_scene_cut && sceneCutPick) pick = true; // 场景切换强制选中
    if (!pick) return false;

    AVFrame* clone = av_frame_clone(frame);
    if (!clone) return false;
    FSSelected sel;
    sel.frame = clone;
    sel.pts_seconds = pts_sec;
    sel.pts = (frame->pts == AV_NOPTS_VALUE ? frame->best_effort_timestamp : frame->pts);
    sel.index_in_stream = total_decoded_;
    if (cfg_.enable_cache) {
        cache_.push_back(sel);
        TrimCache();
    }

    last_selected_frame_index_ = total_decoded_;
    last_selected_pts_sec_ = pts_sec;
    outSelected = clone;
    return true;
}

bool FrameSelector::SceneCutAllowed(double pts_sec) const {
    if (cfg_.min_frames_between_scene_cut > 0 &&
        (total_decoded_ - last_scene_cut_frame_index_) < cfg_.min_frames_between_scene_cut) return false;
    if (cfg_.min_seconds_between_scene_cut > 0.0 && std::isfinite(last_scene_cut_pts_sec_) &&
        (pts_sec - last_scene_cut_pts_sec_) < cfg_.min_seconds_between_scene_cut) return false;
    return true;
}

std::vector<FSSelected> FrameSelector::FetchBatch(size_t maxBatch) {
    std::vector<FSSelected> batch;
    size_t n = std::min(maxBatch, cache_.size());
    for (size_t i = 0; i < n; ++i) batch.push_back(cache_[i]);
    if (cfg_.remove_after_process) {
        for (size_t i = 0; i < n; ++i) av_frame_free(&cache_[i].frame);
        cache_.erase(cache_.begin(), cache_.begin() + n);
    }
    return batch;
}

void FrameSelector::TrimCache() {
    while (static_cast<int>(cache_.size()) > cfg_.max_cached) {
        av_frame_free(&cache_.front().frame);
        cache_.pop_front();
    }
}

void FrameSelector::ClearCache() {
    for (auto& s : cache_) {
        av_frame_free(&s.frame);
    }
    cache_.clear();
}

