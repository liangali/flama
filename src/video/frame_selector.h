#pragma once
// FrameSelector.h
// 抽帧策略与缓存：位于 Decode 与 Scale 之间

#include <deque>
#include <string>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <cmath>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#include "../utils/util.h"
// 选中帧缓存条目
struct FSSelected {
    AVFrame* frame = nullptr;  // 克隆的帧（引用底层数据）
    double pts_seconds = 0.0;  // 时间戳秒
    int64_t pts = 0;           // 原始 pts 或 best_effort_timestamp
    int index_in_stream = 0;   // 全局解码序号
};

// 抽帧器：解码后调用 AcceptDecodedFrame 做决策；被选中后克隆并加入缓存
class FrameSelector {
public:
    explicit FrameSelector(const FSConfig& cfg) : cfg_(cfg) { Reset(); }
    ~FrameSelector() { ClearCache(); }

    void Reset();

    // 解码后调用；返回 true 表示选中，并提供克隆帧 outSelected（供后续 Scale 使用）
    bool AcceptDecodedFrame(AVFrame* frame, AVRational time_base, AVFrame*& outSelected);

    // 通知场景切换（外部可调用：例如视频分割或场景检测模块识别到 scene cut）
    void NotifySceneCut() { pending_scene_cut_ = true; }

    // 拉取一批缓存（不一定在当前需求中使用）
    std::vector<FSSelected> FetchBatch(size_t maxBatch);

    const std::deque<FSSelected>& GetCache() const { return cache_; }

private:
    static double ExtractPtsSec(AVFrame* f, AVRational tb);
    bool DecidePick(double pts_sec);
    bool SceneCutAllowed(double pts_sec) const;
    void TrimCache();
    void ClearCache();

private:
    FSConfig cfg_;
    int total_decoded_ = 0;
    int last_selected_frame_index_ = -INT_MAX;
    double last_selected_pts_sec_ = -DBL_MAX;
    double window_start_sec_ = -DBL_MAX;
    int window_selected_count_ = 0;
    std::deque<FSSelected> cache_;
    // SceneCut 状态
    bool pending_scene_cut_ = false;
    int last_scene_cut_frame_index_ = -INT_MAX;
    double last_scene_cut_pts_sec_ = -DBL_MAX;
};

