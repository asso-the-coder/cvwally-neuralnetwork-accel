#include <cstdio>
#include "tensorflow/lite/micro/debug_log.h"

// Newlib's _write routes these messages to the bare-metal HTIF console.
extern "C" void DebugLog(const char* format, va_list args) {
    std::vfprintf(stderr, format, args);
}
extern "C" int DebugVsnprintf(char* buffer, size_t size, const char* format,
                             va_list args) {
    return std::vsnprintf(buffer, size, format, args);
}
