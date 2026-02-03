//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#pragma once

// Forward declarations
struct FSConfig;
struct BatchConfig;
struct CommonConfig;
struct VLMConfig;

// Logging helpers for runtime global configs
void LogFSConfig(const FSConfig& c);
void LogBatchConfig(const BatchConfig& c);
void LogCommonConfig(const CommonConfig& c);
void LogVLMConfig(const VLMConfig& c);
