//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#include "path_utils.h"
#include <algorithm>
#include <cctype>
#include <cwctype>

#ifdef _WIN32
#include <windows.h>
#include <stdlib.h>
#endif

std::wstring ExtractFilenameWithoutExt(const std::wstring &fullPath)
{
#ifdef _WIN32
    wchar_t drive[_MAX_DRIVE];
    wchar_t dir[_MAX_DIR];
    wchar_t fname[_MAX_FNAME];
    wchar_t ext[_MAX_EXT];

    _wsplitpath_s(fullPath.c_str(), drive, _MAX_DRIVE, dir, _MAX_DIR, fname, _MAX_FNAME, ext, _MAX_EXT);

    return std::wstring(fname);
#else
    // Linux fallback: use filesystem
    std::filesystem::path p(fullPath);
    return p.stem().wstring();
#endif
}

std::string ToLowerAscii(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::wstring ToLowerWide(std::wstring s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
    return s;
}

bool IsVideoExtension(const std::filesystem::path &p)
{
    if (!p.has_extension())
        return false;
#ifdef _WIN32
    std::wstring ext = ToLowerWide(p.extension().wstring());
    return ext == L".mp4" || ext == L".mkv" || ext == L".avi" || ext == L".mov" || ext == L".wmv" || ext == L".flv" || ext == L".webm" || ext == L".m4v";
#else
    std::string ext = ToLowerAscii(p.extension().string());
    return ext == ".mp4" || ext == ".mkv" || ext == ".avi" || ext == ".mov" || ext == ".wmv" || ext == ".flv" || ext == ".webm" || ext == ".m4v";
#endif
}

std::vector<std::filesystem::path> CollectVideoFiles(const std::filesystem::path &dir)
{
    std::vector<std::filesystem::path> files;
    try
    {
        if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir))
            return files;
        for (const auto &entry : std::filesystem::directory_iterator(dir))
        {
            if (!entry.is_regular_file())
                continue;
            const auto &p = entry.path();
            if (IsVideoExtension(p))
                files.push_back(p);
        }
    }
    catch (const std::exception &)
    {
        return files;
    }
    std::sort(files.begin(), files.end(), [](const std::filesystem::path &a, const std::filesystem::path &b) {
#ifdef _WIN32
        return ToLowerWide(a.filename().wstring()) < ToLowerWide(b.filename().wstring());
#else
        return ToLowerAscii(a.filename().string()) < ToLowerAscii(b.filename().string());
#endif
    });
    return files;
}
