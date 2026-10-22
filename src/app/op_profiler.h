#ifndef OP_PROFILER_H
#define OP_PROFILER_H

#include <cstdio>
#include "src/runtime/instret.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"

class OpProfiler final : public tflite::MicroProfilerInterface {
 public:
    uint32_t BeginEvent(const char* tag) override {
        if (count_ == kCapacity) { overflow_ = true; return kCapacity; }
        const uint32_t handle = count_++;
        events_[handle].tag = tag;
        events_[handle].count = read_instret();
        return handle;
    }
    void EndEvent(uint32_t handle) override {
        const uint64_t end = read_instret();
        if (handle < kCapacity) events_[handle].count = end - events_[handle].count;
    }
    bool Print() const {
        if (overflow_) return false;
        for (uint32_t i = 0; i < count_; ++i)
            std::printf("op=%u type=%s instret=%lu\n", i, events_[i].tag,
                        static_cast<unsigned long>(events_[i].count));
        return true;
    }
 private:
    static constexpr uint32_t kCapacity = 128;
    struct Event { const char* tag; uint64_t count; };
    Event events_[kCapacity]{};
    uint32_t count_ = 0;
    bool overflow_ = false;
};
#endif
