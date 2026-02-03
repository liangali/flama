//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#pragma once

#include <thread>
#include <atomic>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

// Synchronization primitives
#ifdef _WIN32
struct CbSync
{
    SRWLOCK lock;
    CONDITION_VARIABLE cv;
    CbSync();
};
CbSync& GetCbSync();
#endif

void CbLock();
void CbUnlock();
void CbNotifyAll();

// Thread functions
void CBLmmEngineThreadFunc();
void CBStatisticsThreadFunc();

// CB Processing functions
void ProcessCBQueueAsync();
void ReportCBQueueAsync();
void ProcessCBQueueAndReport();

// Thread management
void StartCBThreadsIfEnabled();
void StopCBThreadsIfEnabled();
void StartNewCBProcThreadIfEnabled();
void StopNewCBProcThreadIfEnabled();

// Global thread instances
extern std::thread g_lmmEngineThread;
extern std::thread g_statisticsReporterThread;
extern std::thread g_cbProcThread;

// Thread state flags
extern bool cb_threads_started;
extern bool new_cb_proc_running;
extern std::atomic<bool> g_cb_stop;
extern std::atomic<bool> g_cb_finish_flag;
extern std::atomic<bool> finishGenerationThread;

// CB counters
extern int g_cb_pending_batches;
extern int g_cb_enqueued_batches;
extern bool use_cb;
extern int cb_batch_size;
extern int report_idx;

// Timing globals
extern std::chrono::steady_clock::time_point g_hw_cb_start;
extern std::chrono::steady_clock::time_point g_hw_cb_end;
extern std::chrono::steady_clock::time_point g_hw_cb_multi_start;
extern std::chrono::steady_clock::time_point g_hw_cb_multi_end;
extern bool cb_timer;
