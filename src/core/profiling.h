// -----------------------------------------------------------------------------
// Profiling API: per-frame timing, summary, batch aggregation
// -----------------------------------------------------------------------------
#pragma once
#include <atomic>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <cstring>

namespace prof
{

    // Pipeline stages in order
    enum class Stage
    {
        Decode,
        Copy2SharedTex,
        ImportFrameSurface,
        Scale,
        TensorBuild,
        Inference,
        Pipeline
    };

    class FrameProfiler
    {
    public:
        enum
        {
            NumStages = 7
        };
        struct StageRecord
        {
            uint64_t start_us = 0;
            uint64_t end_us = 0;
        };
        struct FrameRecord
        {
            uint64_t frame_idx = 0;
            StageRecord stages[NumStages];
            bool selected = false;
            bool keyframe = false;
            bool scenecut = false;
        };

        FrameProfiler(const FrameProfiler &) = delete;
        FrameProfiler &operator=(const FrameProfiler &) = delete;
        static FrameProfiler &Instance()
        {
            static FrameProfiler inst;
            return inst;
        }
        static FrameProfiler &Get() { return Instance(); }

        void Reset() { std::memset(&cur_, 0, sizeof(cur_)); }
        void BeginFrame(uint64_t idx)
        {
            cur_ = FrameRecord{};
            cur_.frame_idx = idx;
        }
        void MarkStageBegin(Stage st) { cur_.stages[(int)st].start_us = now_us(); }
        void MarkStageEnd(Stage st) { cur_.stages[(int)st].end_us = now_us(); }
        void SetSelectionFlags(bool sel, bool key, bool cut)
        {
            cur_.selected = sel;
            cur_.keyframe = key;
            cur_.scenecut = cut;
        }
        void EndFrameAndWrite();
        static void SetOutputFile(const std::string &path);
        static void SetOutputFileW(const std::wstring &wpath);
        const FrameRecord &Current() const { return cur_; }

    private:
        FrameProfiler() { std::memset(&cur_, 0, sizeof(cur_)); }
        static std::unique_ptr<std::ofstream> &file()
        {
            static std::unique_ptr<std::ofstream> f;
            return f;
        }
        static bool &headerWritten()
        {
            static bool w = false;
            return w;
        }
        static uint64_t now_us()
        {
            using Clock = std::chrono::steady_clock;
            static const auto base = Clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - base).count();
        }
        FrameRecord cur_{};
    };

    struct SummaryCounters
    {
        uint64_t total_frames = 0, selected_frames = 0, keyframes = 0, scenecuts = 0;
    };
    class Summary
    {
    public:
        static Summary &Get()
        {
            static Summary s;
            return s;
        }
        void RecordFrame(bool sel, bool key, bool cut)
        {
            ++c_.total_frames;
            if (sel)
                ++c_.selected_frames;
            if (key)
                ++c_.keyframes;
            if (cut)
                ++c_.scenecuts;
        }
        void PrintSummary() const
        {
            double rate = c_.total_frames ? (double)c_.selected_frames / (double)c_.total_frames : 0.0;
            std::cout << "[Summary] total=" << c_.total_frames << ", selected=" << c_.selected_frames << ", keyframes=" << c_.keyframes << ", scenecuts=" << c_.scenecuts << ", select_rate=" << rate << std::endl;
        }
        const SummaryCounters &counters() const { return c_; }

    private:
        SummaryCounters c_{};
    };

    // External batching: no internal thresholds; caller decides when a batch occurs
    struct BatchRecordLine
    {
        uint64_t batch_index = 0;
        uint64_t decode_frames = 0;
        uint64_t selected_frames = 0;
        uint64_t decode_total_us = 0;
        uint64_t scale_total_us = 0;
        uint64_t tensor_total_us = 0;
        uint64_t inference_us = 0;
        uint64_t pipeline_us = 0;
        std::string prompt;
        std::string result;
    };
    class BatchAggregator
    {
    public:
        static BatchAggregator &Get()
        {
            static BatchAggregator b;
            return b;
        }
        void SetOutputFile(const std::string &path) { openFile(std::filesystem::path(path)); }
        void SetOutputFileW(const std::wstring &wpath) { openFile(std::filesystem::path(wpath)); }
        // Optional: accumulate per-frame to ease batch totals
        void AccumulateFrame(const FrameProfiler::FrameRecord &fr);
        // Caller writes a batch when inference actually happens
        void RecordBatch(uint64_t decodeFrames, uint64_t selectedFrames,
                         uint64_t decodeTotalUs, uint64_t scaleTotalUs, uint64_t tensor_total_us, uint64_t inferenceTotalUs, uint64_t pipeline_us,
                         const std::string &prompt = std::string(), const std::string &result = std::string())
        {
            cur_.batch_index = next_index_++;
            cur_.decode_frames = decodeFrames;
            cur_.selected_frames = selectedFrames;
            cur_.decode_total_us = decodeTotalUs;
            cur_.scale_total_us = scaleTotalUs;
            cur_.tensor_total_us = tensor_total_us;
            cur_.inference_us = inferenceTotalUs;
            cur_.pipeline_us = pipeline_us;
            cur_.prompt = prompt;
            cur_.result = result;
            EmitBatch();
            ResetCurrent();
        }
        // Convenience: record whatever has been accumulated so far as one batch
        void RecordAccumulatedBatch()
        {
            // Use current totals and write one line, then reset
            EmitBatch();
            ResetCurrent();
        }

    private:
        BatchAggregator() { ResetCurrent(); }
        void ResetCurrent()
        {
            cur_ = BatchRecordLine{};
            cur_.batch_index = next_index_ /*++*/;
        }
        void openFile(const std::filesystem::path &p)
        {
            batchFile_.reset();
            bool needHeader = false;
            std::ifstream test(p);
            if (!test.good())
                needHeader = true;
            test.close();
            // Open in binary to ensure exact UTF-8 bytes are written
            auto f = std::make_unique<std::ofstream>(p, std::ios::app | std::ios::binary);
            if (!f->is_open())
            {
                std::cerr << "[BatchProfiler] Failed to open file: " << p.string() << std::endl;
                return;
            }
            batchFile_ = std::move(f);
            if (needHeader)
            {
                // Write UTF-8 BOM to indicate encoding
                const unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
                batchFile_->write(reinterpret_cast<const char *>(bom), 3);
                (*batchFile_) << "batch_index,decode_frames,selected_frames,decode_total_us,decode_total_ms,scale_total_us,scale_total_ms,tensor_total_us,tensor_total_ms, inference_us,inference_ms,pipeline_us,pipeline_ms,prompt,result\n";
            }
        }
        static std::string SanitizeCSV(const std::string &in)
        {
            std::string out;
            out.reserve(in.size());
            for (char c : in)
            {
                if (c == '\n' || c == '\r' || c == ',')
                    out.push_back(' ');
                else
                    out.push_back(c);
            }
            return out;
        }
        void EmitBatch()
        {
            if (!batchFile_ || !batchFile_->is_open())
                return;
            (*batchFile_) << cur_.batch_index << ','
                          << cur_.decode_frames << ','
                          << cur_.selected_frames << ','
                          << cur_.decode_total_us << ','
                          << (cur_.decode_total_us / 1000.0) << ','
                          << cur_.scale_total_us << ','
                          << (cur_.scale_total_us / 1000.0) << ','
                          << cur_.tensor_total_us << ','
                          << (cur_.tensor_total_us / 1000.0) << ','
                          << cur_.inference_us << ','
                          << (cur_.inference_us / 1000.0) << ','
                          << cur_.pipeline_us << ','
                          << (cur_.pipeline_us / 1000.0) << ','
                          << SanitizeCSV(cur_.prompt) << ','
                          << SanitizeCSV(cur_.result) << '\n';
            batchFile_->flush();
        }
        BatchRecordLine cur_{};
        uint64_t next_index_ = 0;
        std::unique_ptr<std::ofstream> batchFile_{};
    };

    // ---- Inline method implementations that depend on class layout ----
    inline void FrameProfiler::SetOutputFile(const std::string &path)
    {
        file().reset();
        bool needBOM = false;
        std::ifstream test(path, std::ios::binary);
        if (!test.good())
            needBOM = true;
        test.close();
        auto f = std::make_unique<std::ofstream>(path, std::ios::app | std::ios::binary);
        if (!f->is_open())
        {
            std::cerr << "[FrameProfiler] Failed to open file: " << path << std::endl;
            return;
        }
        if (needBOM)
        {
            const unsigned char bom[] = {0xEF, 0xBB, 0xBF};
            f->write(reinterpret_cast<const char *>(bom), 3);
        }
        file() = std::move(f);
        headerWritten() = false;
    }
    inline void FrameProfiler::SetOutputFileW(const std::wstring &wpath)
    {
        std::filesystem::path p(wpath);
        file().reset();
        bool needBOM = false;
        std::ifstream test(p, std::ios::binary);
        if (!test.good())
            needBOM = true;
        test.close();
        auto f = std::make_unique<std::ofstream>(p, std::ios::app | std::ios::binary);
        if (!f->is_open())
        {
            std::wcerr << L"[FrameProfiler] Failed to open file: " << wpath << std::endl;
            return;
        }
        if (needBOM)
        {
            const unsigned char bom[] = {0xEF, 0xBB, 0xBF};
            f->write(reinterpret_cast<const char *>(bom), 3);
        }
        file() = std::move(f);
        headerWritten() = false;
    }
    inline void FrameProfiler::EndFrameAndWrite()
    {
        auto &f = file();
        if (!f || !f->is_open())
            return;
        if (!headerWritten())
        {
            *f << "frame,stage,begin_us,end_us,duration_us,duration_ms,selected,keyframe,scenecut\n";
            headerWritten() = true;
        }
        static const char *names[NumStages] = {"Decode", "Copy2SharedTex", "ImportFrameSurface", "Scale", "TensorBuild", "Inference", "Pipeline"};
        for (int i = 0; i < NumStages; ++i)
        {
            const auto &rec = cur_.stages[i];
            uint64_t dur = (rec.end_us > rec.start_us) ? (rec.end_us - rec.start_us) : 0;
            *f << cur_.frame_idx << ',' << names[i] << ',' << rec.start_us << ',' << rec.end_us << ',' << dur << ',' << (dur / 1000.0) << ',' << (cur_.selected ? 1 : 0) << ',' << (cur_.keyframe ? 1 : 0) << ',' << (cur_.scenecut ? 1 : 0) << '\n';
        }
        f->flush();
        // No auto-batch here; caller can use BatchAggregator::AccumulateFrame if desired
    }
    inline void BatchAggregator::AccumulateFrame(const FrameProfiler::FrameRecord &fr)
    {
        const auto &dec = fr.stages[(int)Stage::Decode];
        const auto &scl = fr.stages[(int)Stage::Scale];
        const auto &inf = fr.stages[(int)Stage::Inference];
        uint64_t dDur = dec.end_us > dec.start_us ? dec.end_us - dec.start_us : 0;
        uint64_t sDur = scl.end_us > scl.start_us ? scl.end_us - scl.start_us : 0;
        uint64_t iDur = inf.end_us > inf.start_us ? inf.end_us - inf.start_us : 0;
        cur_.decode_frames++;
        cur_.decode_total_us += dDur;
        if (fr.selected)
        {
            cur_.selected_frames++;
            cur_.scale_total_us += sDur;
            cur_.tensor_total_us += sDur;
            cur_.inference_us += iDur;
        }
    }

} // namespace prof
