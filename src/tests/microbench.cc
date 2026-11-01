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
#ifdef CVW_RQNT
int32_t accumulators[1280], multipliers[1280], shifts[1280];
uint64_t params[1280];
int8_t scalar_outputs[1280], rqnt_outputs[1280];
__attribute__((noinline)) void ScalarRequant(int count, int zp, int lo, int hi) {
  for (int i = 0; i < count; ++i) {
    int32_t value = tflite::MultiplyByQuantizedMultiplier(accumulators[i], multipliers[i], shifts[i]);
    value += zp;
    value = std::max(lo, std::min(hi, value));
    scalar_outputs[i] = static_cast<int8_t>(value);
  }
}
__attribute__((noinline)) void PackedRequant(int count) {
  for (int i = 0; i < count; ++i)
    rqnt_outputs[i] = static_cast<int8_t>(cvw_rqnt(accumulators[i], params[i]));
}
#endif

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
#ifdef CVW_RQNT
  for (int i = 0; i < 1280; ++i) {
    seed = seed * 1664525u + 1013904223u;
    accumulators[i] = static_cast<int32_t>(seed % 65537) - 32768;
    multipliers[i] = 0x40000000 + (seed & 0x3fffffff);
    shifts[i] = -2 - (i % 7);
    params[i] = cvw_rqnt_pack(multipliers[i], shifts[i], -7, -128, 127);
  }
  const uint64_t e = read_instret();
  ScalarRequant(1280, -7, -128, 127);
  const uint64_t f = read_instret();
  const uint64_t g = read_instret();
  PackedRequant(1280);
  const uint64_t h = read_instret();
  for (int i = 0; i < 1280; ++i) {
    if (scalar_outputs[i] != rqnt_outputs[i]) {
      std::printf("FAIL: requant microbench index=%d\n", i);
      return 1;
    }
  }
  std::printf("microbench,requant_scalar,1280,%lu\nmicrobench,rqnt,1280,%lu\n",
              static_cast<unsigned long>(f-e), static_cast<unsigned long>(h-g));
  std::printf("PASS: requant microbench\n");
#endif
  return 0;
}
