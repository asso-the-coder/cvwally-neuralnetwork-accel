#include <cstdio>
#include "src/kernels/xqmac.h"
#include "build/xqmac/rqnt_vectors.h"
#ifndef CVW_RQNT
#error "Instruction tests require CVW_RQNT"
#endif
int main() {
  unsigned count = 0;
  for (const auto& v : rqnt_vectors) {
    const int64_t actual = cvw_rqnt(static_cast<int64_t>(v.acc), v.params);
    const int64_t fallback = cvw_rqnt_c(static_cast<int64_t>(v.acc), v.params);
    const uint64_t repacked = cvw_rqnt_pack(static_cast<int32_t>(v.params),
        static_cast<int8_t>(v.params >> 32), static_cast<int8_t>(v.params >> 40),
        static_cast<int8_t>(v.params >> 48), static_cast<int8_t>(v.params >> 56));
    int64_t zero;
    __asm__ volatile(".insn r 0x0B, 1, 0, zero, %1, %2\n\taddi %0, zero, 0"
                     : "=r"(zero) : "r"(v.acc), "r"(v.params));
    if (actual != v.expected || fallback != v.expected || zero != 0 || repacked != v.params) {
      std::printf("FAIL: rqnt vector=%u actual=%ld fallback=%ld expected=%ld\n",
                  count, static_cast<long>(actual), static_cast<long>(fallback),
                  static_cast<long>(v.expected));
      return 1;
    }
    ++count;
  }
  std::printf("PASS: rqnt directed=%u random=%u total=%u (instruction, C, packing, rd=x0)\n",
              rqnt_directed_count, count - rqnt_directed_count, count);
  return 0;
}
