//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#pragma once

#include <string>
#include <vector>
#include <filesystem>

// Extract filename without extension from a full path (Windows-specific)
std::wstring ExtractFilenameWithoutExt(const std::wstring &fullPath);

// Convert string to lowercase (ASCII)
std::string ToLowerAscii(std::string s);

// Convert wide string to lowercase
std::wstring ToLowerWide(std::wstring s);

// Check if a path has a video file extension
bool IsVideoExtension(const std::filesystem::path &p);

// Collect all video files from a directory, sorted by filename
std::vector<std::filesystem::path> CollectVideoFiles(const std::filesystem::path &dir);
