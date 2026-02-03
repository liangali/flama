//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "segment_timing.h"
#include "batch/batch_state.h"
#include "output/vlm_json_output.h"
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
    if (!g_batchState.seg_has_pts)
    {
        g_batchState.seg_start_sec = pts_sec;
        g_batchState.seg_end_sec = pts_sec;
        g_batchState.seg_has_pts = true;
        return;
    }
    if (pts_sec < g_batchState.seg_start_sec)
        g_batchState.seg_start_sec = pts_sec;
    if (pts_sec > g_batchState.seg_end_sec)
        g_batchState.seg_end_sec = pts_sec;
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
