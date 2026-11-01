#ifndef CVW_XQMAC_H
#define CVW_XQMAC_H
#include <stdint.h>

static inline int64_t cvw_qmac8_c(uint64_t a, uint64_t b) {
  int64_t sum = 0;
  for (unsigned i = 0; i < 8; ++i) {
    const int av = (a >> (8 * i)) & 255;
    const int bv = (b >> (8 * i)) & 255;
    sum += (av < 128 ? av : av - 256) * (bv < 128 ? bv : bv - 256);
  }
  return sum;
}

static inline int64_t cvw_qmac8(uint64_t a, uint64_t b) {
#ifdef CVW_XQMAC
  int64_t result;
  __asm__ volatile(".insn r 0x0B, 0, 0, %0, %1, %2" : "=r"(result) : "r"(a), "r"(b));
  return result;
#else
  return cvw_qmac8_c(a, b);
#endif
}

#ifdef CVW_XQMAC
// Call only after checking the layer bases and strides. memcpy avoids aliasing UB.
static inline uint64_t cvw_load8(const int8_t* ptr) {
  uint64_t value;
  __builtin_memcpy(&value, __builtin_assume_aligned(ptr, 8), sizeof(value));
  return value;
}
#ifdef __cplusplus
namespace cvw {
extern unsigned qmac8_layers, scalar_layers;
}
#endif
#endif
#ifdef __cplusplus
#include "tensorflow/lite/micro/kernels/internal/common.h"

static inline uint64_t cvw_rqnt_pack(int32_t mult, int shift, int zp, int lo, int hi) {
  return static_cast<uint32_t>(mult) | (uint64_t(uint8_t(shift)) << 32) |
         (uint64_t(uint8_t(zp)) << 40) | (uint64_t(uint8_t(lo)) << 48) |
         (uint64_t(uint8_t(hi)) << 56);
}

static inline int64_t cvw_rqnt_c(int64_t acc, uint64_t params) {
  const int32_t q = tflite::MultiplyByQuantizedMultiplier(
      static_cast<int32_t>(acc), static_cast<int32_t>(params),
      static_cast<int8_t>(params >> 32));
  // Unsigned addition makes the specified int32 wrap explicit.
  const int32_t y = static_cast<int32_t>(static_cast<uint32_t>(q) +
      static_cast<uint32_t>(static_cast<int8_t>(params >> 40)));
  const int32_t lo = static_cast<int8_t>(params >> 48);
  const int32_t hi = static_cast<int8_t>(params >> 56);
  return std::min(std::max(y, lo), hi);
}

static inline int64_t cvw_rqnt(int64_t acc, uint64_t params) {
#ifdef CVW_RQNT
  int64_t result;
  __asm__ volatile(".insn r 0x0B, 1, 0, %0, %1, %2" : "=r"(result) : "r"(acc), "r"(params));
  return result;
#else
  return cvw_rqnt_c(acc, params);
#endif
}
#endif
#endif
