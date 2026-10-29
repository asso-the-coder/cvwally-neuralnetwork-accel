#include <cstdio>
#include "src/kernels/xqmac.h"
#include "build/xqmac/vectors.h"
#ifndef CVW_XQMAC
#error "Instruction tests require CVW_XQMAC"
#endif
int main() {
  unsigned count = 0;
  for (const auto& v : vectors) {
    const int64_t actual = cvw_qmac8(v.a, v.b);
    const int64_t fallback = cvw_qmac8_c(v.a, v.b);
    int64_t square, zero;
    // Force identical source register indices, then verify architectural x0.
    __asm__ volatile(".insn r 0x0B, 0, 0, %0, %1, %1" : "=r"(square) : "r"(v.a));
    __asm__ volatile(".insn r 0x0B, 0, 0, zero, %1, %2\n\taddi %0, zero, 0"
                     : "=r"(zero) : "r"(v.a), "r"(v.b));
    if (actual != v.expected || fallback != v.expected || square != v.square || zero != 0) {
      std::printf("FAIL: qmac8 vector=%u actual=%ld fallback=%ld expected=%ld\n",
                  count, static_cast<long>(actual), static_cast<long>(fallback), static_cast<long>(v.expected));
      return 1;
    }
    ++count;
  }
  std::printf("PASS: qmac8 directed=%u random=%u total=%u (instruction, C, rs1=rs2, rd=x0)\n",
              directed_count, count - directed_count, count);
  return 0;
}
