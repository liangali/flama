//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "config_logging.h"
#include "../utils/util.h"
#include "../utils/debug.h"

void LogFSConfig(const FSConfig& c)
{
    DBG_LOGF("[FSConfig] policy=%d frame_interval=%d window_seconds=%.3f max_per_window=%d min_frames_between=%d min_seconds_between=%.3f max_cached=%d remove_after_process=%d force_keyframe=%d enable_scene_cut=%d min_frames_between_scene_cut=%d min_seconds_between_scene_cut=%.3f enable_cache=%d",
        (int)c.policy,
        c.frame_interval,
        c.window_seconds,
        c.max_per_window,
        c.min_frames_between,
        c.min_seconds_between,
        c.max_cached,
        c.remove_after_process ? 1 : 0,
        c.force_keyframe ? 1 : 0,
        c.enable_scene_cut ? 1 : 0,
        c.min_frames_between_scene_cut,
        c.min_seconds_between_scene_cut,
        c.enable_cache ? 1 : 0);
}

void LogBatchConfig(const BatchConfig& c)
{
    DBG_LOGF("[BatchConfig] new_batch_mode=%d batch_trigger=%d max_cache=%d decode_window=%d flush_partial=%d cb_batch_size=%d",
        c.new_batch_mode ? 1 : 0,
        c.batch_trigger,
        c.max_cache,
        c.decode_window,
        c.flush_partial ? 1 : 0,
        c.cb_batch_size);
}

void LogCommonConfig(const CommonConfig& c)
{
    DBG_LOGF("[CommonConfig] debug=%d use_cb=%d cb_multi_thread=%d new_multithread=%d hw_decode=%d input_video_path=%s vpp_down=%dx%d",
        c.debug ? 1 : 0,
        c.use_cb ? 1 : 0,
        c.cb_multi_thread ? 1 : 0,
        c.new_multithread ? 1 : 0,
        c.hw_decode ? 1 : 0,
        c.input_video_path.c_str(),
        c.vpp_down_width,
        c.vpp_down_height);
}

void LogVLMConfig(const VLMConfig& c)
{
    DBG_LOGF("[VLMConfig] path=%s device=%s enable_continuous_batching=%d",
        c.path.c_str(),
        c.device.c_str(),
        c.enable_continuous_batching ? 1 : 0);
    DBG_LOG(std::string("[VLMConfig] scheduler=") + c.shedulerConfig.to_string());
}
