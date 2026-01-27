#pragma once
#include <atomic>
#include <string>
#include <cstdio>
#include <cstdarg>

// Global debug mode flag (thread-safe)
extern std::atomic<bool> g_isDebugMode;

inline bool IsDebugMode() { return g_isDebugMode.load(std::memory_order_relaxed); }
inline void SetDebugMode(bool v) { g_isDebugMode.store(v, std::memory_order_relaxed); }

// Simple printf-style helper only active in debug mode
inline void DebugPrintf(const char* fmt, ...) {
    if (!IsDebugMode()) return;
    va_list ap; va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    std::fputc('\n', stdout);
    va_end(ap);
}

// Stream-style macro (builds std::string only when enabled)
#define DBG_LOG(msg) do { if (IsDebugMode()) { std::printf("%s\n", (std::string(msg)).c_str()); } } while(0)
#define DBG_LOGF(fmt, ...) do { if (IsDebugMode()) { std::printf((fmt), __VA_ARGS__);std::printf("\n"); } } while(0)

// Scoped override helper (optional)
class DebugScope {
public:
    explicit DebugScope(bool enable) : prev_(IsDebugMode()) { SetDebugMode(enable); }
    ~DebugScope() { SetDebugMode(prev_); }
private:
    bool prev_;
};
