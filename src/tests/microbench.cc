#include <cstdio>
#include "src/kernels/xqmac.h"
#include "src/runtime/instret.h"
#ifndef CVW_XQMAC
#error "Microbenchmark requires CVW_XQMAC"
#endif
namespace {
alignas(16) int8_t weights[1280], input[1280];
__attribute__((noinline)) int32_t Scalar(const int8_t* wt, const int8_t* x, int depth) {
  int32_t acc = 0;
  for (int c = 0; c < depth; ++c) acc += static_cast<int32_t>(*wt++) * *x++;
  return acc;
}
__attribute__((noinline)) int32_t Packed(const int8_t* wt, const int8_t* x, int depth) {
  int32_t acc = 0;
  int c = 0;
  for (; c + 8 <= depth; c += 8) {
    acc += cvw_qmac8(cvw_load8(wt), cvw_load8(x));
    wt += 8;
    x += 8;
  }
  for (; c < depth; ++c) acc += static_cast<int32_t>(*wt++) * *x++;
  return acc;
}
}
int main() {
  uint32_t seed = 0x514d4143;
  for (int i = 0; i < 1280; ++i) {
    seed = seed * 1664525u + 1013904223u;
    weights[i] = static_cast<int8_t>(static_cast<int>(seed >> 24) - 128);
    seed = seed * 1664525u + 1013904223u;
    input[i] = static_cast<int8_t>(static_cast<int>(seed >> 24) - 128);
  }
  const uint64_t a = read_instret();
  const int32_t scalar = Scalar(weights, input, 1280);
  // Keep the returned value inside the measured region even under optimization.
  __asm__ volatile("" : : "r"(scalar) : "memory");
  const uint64_t b = read_instret();
  const uint64_t c = read_instret();
  const int32_t packed = Packed(weights, input, 1280);
  __asm__ volatile("" : : "r"(packed) : "memory");
  const uint64_t d = read_instret();
  if (scalar != packed) { std::printf("FAIL: microbench results differ\n"); return 1; }
  std::printf("microbench,scalar,1280,%lu\nmicrobench,qmac8,1280,%lu\n",
              static_cast<unsigned long>(b-a), static_cast<unsigned long>(d-c));
  std::printf("PASS: microbench result=%d\n", scalar);
  return 0;
}
