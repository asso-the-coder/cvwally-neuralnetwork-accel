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
#endif
