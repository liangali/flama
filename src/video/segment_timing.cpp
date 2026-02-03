//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "segment_timing.h"
#include "batch/batch_state.h"
#include "json/vlm_json_output.h"
#include "utils/util.h"
#include <cmath>

double RoundTimeSec(double v)
{
    return std::round(v * 1000.0) / 1000.0;
}

double GetFramePtsSeconds(const AVFrame *frame, AVRational time_base)
{
    if (!frame)
        return 0.0;
    int64_t pts = (frame->pts == AV_NOPTS_VALUE ? frame->best_effort_timestamp : frame->pts);
    if (pts == AV_NOPTS_VALUE)
        pts = 0;
    return av_q2d(time_base) * static_cast<double>(pts);
}

void UpdateBatchSegmentTiming(double pts_sec)
{
    // Calculate time window end boundary for proper segment duration
    double window_seconds = g_fsConfig.window_seconds;
    double window_end = pts_sec;
    if (window_seconds > 0.0) {
        // Calculate the end boundary of the time window containing this frame
        // Add small epsilon to handle exact boundary cases (e.g., pts_sec = 1.0 should be in window [1, 2))
        window_end = std::ceil((pts_sec + 0.0001) / window_seconds) * window_seconds;
    }

    if (!g_batchState.seg_has_pts)
    {
        // Use next_seg_start_sec for segment continuity
        // For the first segment, next_seg_start_sec is initialized to 0.0
        g_batchState.seg_start_sec = g_batchState.next_seg_start_sec;
        g_batchState.seg_end_sec = window_end;
        g_batchState.seg_has_pts = true;
        return;
    }
    // Only update seg_end_sec (seg_start_sec is fixed at segment start)
    if (window_end > g_batchState.seg_end_sec)
        g_batchState.seg_end_sec = window_end;
}

void RecordJsonSegmentFromTimes(double start_sec, double end_sec, const std::string &desc)
{
    g_vlmJsonCollector.AddSegment(RoundTimeSec(start_sec), RoundTimeSec(end_sec), desc);
}

void RecordJsonSegmentFromBatch(const BatchState &bs, const std::string &desc)
{
    if (bs.seg_has_pts)
        RecordJsonSegmentFromTimes(bs.seg_start_sec, bs.seg_end_sec, desc);
    else
        RecordJsonSegmentFromTimes(0.0, 0.0, desc);
}
