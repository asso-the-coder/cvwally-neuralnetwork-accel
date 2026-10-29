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
    return {{match, mask, Qmac8, Qmac8, Qmac8, Qmac8, Qmac8, Qmac8, Qmac8, Qmac8}};
  }
  std::vector<disasm_insn_t*> get_disasms(const processor_t*) override {
    return {new disasm_insn_t("qmac8", match, mask, {&rd, &rs1, &rs2})};
  }
};
}
REGISTER_EXTENSION(xqmac, []() { return new Xqmac; })
