# Xqmac: custom instructions for INT8 inference

Design constraint: every instruction is an ordinary R-type instruction with two source registers and one
destination. Nothing is added to the register file or architectural state (no extra ports, CSRs or hidden
accumulators), so in a pipeline each instruction is a new function unit in the execute stage, and decode,
hazard and forwarding logic stay as for any other ALU op.

## qmac8: packed INT8 dot product

```
qmac8 rd, rs1, rs2
```

| field  | funct7  | rs2 | rs1 | funct3 | rd | opcode            |
|--------|---------|-----|-----|--------|----|-------------------|
| bits   | 31:25   | 24:20 | 19:15 | 14:12 | 11:7 | 6:0           |
| value  | 0000000 | rs2 | rs1 | 000    | rd | 0001011 (custom-0) |

Semantics (RV64):

```
sum = 0
for i in 0..7:
    a = sign_extend(rs1[8*i+7 : 8*i])     # int8
    b = sign_extend(rs2[8*i+7 : 8*i])     # int8
    sum += a * b
rd = sign_extend_64(sum)
```

- Lane i is byte i of a little-endian 64-bit load, so `ld` of eight consecutive int8 values feeds the lanes
  in memory order.
- The sum is in [-130048, 131072] (worst case 8 x (-128) x (-128)), so it is always exact: no overflow, wrap
  or saturation case exists.
- No traps, no flags, no state. `rd = x0` discards the result.
- Accumulation is a separate `add`: `acc += qmac8(w, x)`. A fused accumulate would need `rd` as a third source
  operand, i.e. a third register-file read port.
- Input zero points are not handled by the instruction: the term `input_offset * sum(w)` is constant per
  output channel and is folded into the bias once at model setup, so the lanes only see plain int8 values.

C wrapper (`src/kernels/xqmac.h`): `int64_t cvw_qmac8(uint64_t w, uint64_t x)`, emitted with
`.insn r 0x0B, 0, 0, rd, rs1, rs2`, with a pure-C fallback of identical semantics when the extension is off.

## rqnt: requantization

```
rqnt rd, rs1, rs2
```

| field  | funct7  | rs2 | rs1 | funct3 | rd | opcode            |
|--------|---------|-----|-----|--------|----|-------------------|
| value  | 0000000 | rs2 | rs1 | 001    | rd | 0001011 (custom-0) |

Operands:

| bits of rs2 | field | type |
|---|---|---|
| 31:0  | multiplier `M` (Q31, >= 0) | int32 |
| 39:32 | shift `s` (-31 ... 30; > 0 = left) | int8 |
| 47:40 | output zero point `zp` | int8 |
| 55:48 | clamp minimum `lo` | int8 |
| 63:56 | clamp maximum `hi` | int8 |

`rs1[31:0]` is the int32 accumulator (bias already added); `rs1[63:32]` is ignored.

Semantics: the TFLM/gemmlowp reference `MultiplyByQuantizedMultiplier` (double rounding), then offset and clamp:

```
x  = int32(rs1[31:0])
l  = max(s, 0);  r = max(-s, 0)
x  = int32(x << l)                                   # 32-bit wrap
p  = int64(x) * int64(M)
if x == M == INT32_MIN:  h = INT32_MAX               # saturation case
else:                    h = int32((p + (p >= 0 ? 2^30 : 1 - 2^30)) / 2^31)   # truncating division
mask = 2^r - 1;  rem = h & mask
thr  = (mask >> 1) + (h < 0 ? 1 : 0)
q  = (h >> r) + (rem > thr ? 1 : 0)                  # round half away from zero
y  = int32(q + zp)                                   # 32-bit wrap, as the reference int32 add
y  = min(max(y, lo), hi)
rd = sign_extend_64(y)
```

- The parameter word is constant per output channel (`M`, `s` per channel; `zp`, `lo`, `hi` per layer), so it is
  packed once at model setup into an 8-byte table and loaded with one `ld` per output value.
- If `lo > hi` the result is `hi` (the clamp order above); TFLM never produces this case.
- No traps, no flags, no state.

C wrapper: `int64_t cvw_rqnt(int64_t acc, uint64_t params)` (`.insn r 0x0B, 1, 0, rd, rs1, rs2`) plus a pure-C
fallback built from the same TFLM functions.
