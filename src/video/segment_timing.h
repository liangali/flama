//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#pragma once

#include <string>

extern "C"
{
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
}

// Forward declaration
struct BatchState;

// Round time value to 3 decimal places
double RoundTimeSec(double v);

// Get frame PTS in seconds
double GetFramePtsSeconds(const AVFrame *frame, AVRational time_base);

// Update batch segment timing with a new PTS value
void UpdateBatchSegmentTiming(double pts_sec);

// Record a JSON segment from explicit start/end times
void RecordJsonSegmentFromTimes(double start_sec, double end_sec, const std::string &desc);

// Record a JSON segment from batch state
void RecordJsonSegmentFromBatch(const BatchState &bs, const std::string &desc);
