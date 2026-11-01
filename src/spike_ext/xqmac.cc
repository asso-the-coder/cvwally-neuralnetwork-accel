#include "extension.h"
#include "trap.h"

namespace {
constexpr reg_t match = 0x0000000b;
constexpr reg_t mask = 0xfe00707f;

reg_t Qmac8(processor_t* p, insn_t insn, reg_t pc) {
  if (p->get_xlen() != 64) throw trap_illegal_instruction(insn.bits());
  auto& state = *p->get_state();
  const reg_t a = state.XPR[insn.rs1()], b = state.XPR[insn.rs2()];
  int64_t sum = 0;
  for (unsigned i = 0; i < 8; ++i) {
    const int av = (a >> (8 * i)) & 255;
    const int bv = (b >> (8 * i)) & 255;
    sum += (av < 128 ? av : av - 256) * (bv < 128 ? bv : bv - 256);
  }
  state.XPR.write(insn.rd(), static_cast<reg_t>(sum));
  if (p->get_log_commits_enabled())
    state.log_reg_write[insn.rd() << 4] = {static_cast<reg_t>(sum), 0};
  return pc + 4;
}

constexpr reg_t rqnt_match = 0x0000100b;
reg_t Rqnt(processor_t* p, insn_t insn, reg_t pc) {
  if (p->get_xlen() != 64) throw trap_illegal_instruction(insn.bits());
  auto& state = *p->get_state();
  const uint64_t params = state.XPR[insn.rs2()];
  const int shift = static_cast<int8_t>(params >> 32);
  int32_t y = 0;
  // Reserved shifts have no specified result; avoid undefined host shifts.
  if (shift >= -31 && shift <= 30) {
    const int32_t x = static_cast<int32_t>(
        static_cast<uint32_t>(state.XPR[insn.rs1()]) << (shift > 0 ? shift : 0));
    const int32_t m = static_cast<int32_t>(params);
    const int64_t product = static_cast<int64_t>(x) * m;
    const int32_t h = x == INT32_MIN && m == INT32_MIN ? INT32_MAX :
        static_cast<int32_t>((product + (product >= 0 ? (INT64_C(1) << 30) :
                                       1 - (INT64_C(1) << 30))) / (INT64_C(1) << 31));
    const int right = shift < 0 ? -shift : 0;
    const uint32_t bits = (UINT32_C(1) << right) - 1;
    const uint32_t threshold = (bits >> 1) + (h < 0);
    const int32_t q = (h >> right) + ((static_cast<uint32_t>(h) & bits) > threshold);
    y = static_cast<int32_t>(static_cast<uint32_t>(q) +
                            static_cast<uint32_t>(static_cast<int8_t>(params >> 40)));
    const int32_t lo = static_cast<int8_t>(params >> 48), hi = static_cast<int8_t>(params >> 56);
    y = std::min(std::max(y, lo), hi);
  }
  const reg_t result = static_cast<reg_t>(static_cast<int64_t>(y));
  state.XPR.write(insn.rd(), result);
  if (p->get_log_commits_enabled()) state.log_reg_write[insn.rd() << 4] = {result, 0};
  return pc + 4;
}

struct RegisterArg : arg_t {
  explicit RegisterArg(unsigned field) : field(field) {}
  std::string to_string(insn_t insn) const override {
    return xpr_name[field == 0 ? insn.rd() : field == 1 ? insn.rs1() : insn.rs2()];
  }
  unsigned field;
};
RegisterArg rd(0), rs1(1), rs2(2);

class Xqmac : public extension_t {
 public:
  const char* name() const override { return "xqmac"; }
  std::vector<insn_desc_t> get_instructions(const processor_t&) override {
    return {{match, mask, Qmac8, Qmac8, Qmac8, Qmac8, Qmac8, Qmac8, Qmac8, Qmac8},
            {rqnt_match, mask, Rqnt, Rqnt, Rqnt, Rqnt, Rqnt, Rqnt, Rqnt, Rqnt}};
  }
  std::vector<disasm_insn_t*> get_disasms(const processor_t*) override {
    return {new disasm_insn_t("qmac8", match, mask, {&rd, &rs1, &rs2}),
            new disasm_insn_t("rqnt", rqnt_match, mask, {&rd, &rs1, &rs2})};
  }
};
}
REGISTER_EXTENSION(xqmac, []() { return new Xqmac; })
