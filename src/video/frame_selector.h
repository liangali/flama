#pragma once
// FrameSelector.h
// Frame selection strategy and cache: located between Decode and Scale

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

// Selected frame cache entry
struct FSSelected {
    AVFrame* frame = nullptr;  // Cloned frame (references underlying data)
    double pts_seconds = 0.0;  // Timestamp in seconds
    int64_t pts = 0;           // Original PTS or best_effort_timestamp
    int index_in_stream = 0;   // Global decode index
};

// Frame selector: call AcceptDecodedFrame after decoding to make selection decision
// Selected frames are cloned and added to cache
class FrameSelector {
public:
    explicit FrameSelector(const FSConfig& cfg) : cfg_(cfg) { Reset(); }
    ~FrameSelector() { ClearCache(); }

    void Reset();

    // Called after decoding; returns true if frame selected, provides cloned frame in outSelected (for scaling)
    bool AcceptDecodedFrame(AVFrame* frame, AVRational time_base, AVFrame*& outSelected);

    // Notify scene cut (can be called externally, e.g., by scene detection module)
    void NotifySceneCut() { pending_scene_cut_ = true; }

    // Fetch batch of cached frames (not necessarily used in current request)
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
    // Scene cut state
    bool pending_scene_cut_ = false;
    int last_scene_cut_frame_index_ = -INT_MAX;
    double last_scene_cut_pts_sec_ = -DBL_MAX;
};

