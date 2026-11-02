# Design

This document explains the main design decisions, the alternatives considered, and why each choice was made.
The exact instruction encodings and semantics are in [docs/isa_xqmac.md](docs/isa_xqmac.md); measurements are in
[EXPERIMENTS.md](EXPERIMENTS.md).

## 1. Workload and runtime

**Model: EfficientNet-Lite0 INT8, used unmodified.** It has 4.7 M parameters and 385 M MACs per inference.
Alternatives were MobileNetV2/V3, ResNet-18 and a small Transformer (DeiT-Tiny). The Lite variants were designed
for INT8 (ReLU6, no squeeze-and-excitation), so every operator is a stock TFLite Micro operator:
- 33 CONV_2D
- 16 DEPTHWISE_CONV_2D
- 9 ADD
- 1 FULLY_CONNECTED
- plus pooling, softmax, quantize and reshape

The model is dominated by 1×1 convolutions (88% of MACs), which are dot products over channels, and its channel
counts are multiples of 8. That maps directly onto an 8-lane packed MAC. A Transformer would add INT8
softmax/LayerNorm/GELU porting risk without changing what the project measures.

**Source and pinning.** The official Google tarball is pinned by URL and SHA256. No quantization or retraining
(per course FAQ).

**Runtime: TFLite Micro, bare-metal.** Only the used operators are registered, the tensor arena is static, there is
no `malloc` during inference and no OS. The board support is minimal: `crt0`, linker script, HTIF console and exit.

**Correctness reference: LiteRT with `BUILTIN_REF` kernels.** We measured that LiteRT's default and XNNPACK kernels
round differently from the reference kernels by 1–5 LSB on 6–14 of the 1,000 output bytes. They therefore cannot
serve as a bit-exact reference. With `BUILTIN_REF`, our Spike output matches all 1,000 bytes on every input.

**No floating point.** Spike runs with `--isa=rv64imc`, so any FP instruction would trap, and an objdump scan of
every ELF confirms that none exist. The compiler's soft-float routines run only in one-time setup
(TFLM computing requantization multipliers), never during inference.

## 2. Methodology: one change per configuration

B0 is TFLite Micro exactly as shipped. Every later optimization is its own configuration, measured with the same
scripts, model and inputs, so each speedup can be traced to a single change:

```
B0 ─► B1a ─► B1b ─► B1c ─► X1 ─► X2
      (software only)      (+qmac8) (+rqnt)
```

The software steps (B1) come first, on purpose. A custom instruction should be justified against the best plain
RV64IMC code, not against an unoptimized baseline.

## 3. Software optimizations (B1)

| Config | Change | Why | Alternative rejected |
|---|---|---|---|
| B1a | Release build (`-DNDEBUG`) | In B0's conv inner loop, 37 of 59 instructions per MAC were debug-only index checks (`TFLITE_DCHECK`) | Calling B0 "the release build", which would hide a 4.9× factor inside the baseline |
| B1b | Depthwise: precompute interior/border ranges; the interior loop has no padding checks | Whether a filter window crosses the image edge is known before the loop starts | im2col (copy each window into a contiguous buffer): in depthwise every input byte feeds only one MAC per channel, so the copy costs as much as it saves, plus 9–25× memory |
| B1c | Fold the input zero-point term into the bias (1×1 conv, FC) | Σw·(x+o) = Σw·x + o·Σw, and o·Σw is constant per output channel, so it is precomputed once at setup | Forcing zero point −128 everywhere: needs re-quantizing the model (out of scope) and breaks bit-exactness |

B1c also prepares for `qmac8`. Without the folding, (x + o) spans 0…255 and does not fit a signed byte lane.

## 4. Custom instructions (Xqmac)

### Design constraint
The professor's guidance for this project: leave the register file and the rest of the core as-is. So both
instructions are **standard R-type: two source registers, one destination, no new architectural state, no CSRs.**
To the pipeline they look like an `add`. All new logic sits in the execute stage, and decode, hazards, forwarding
and writeback are unchanged. This also keeps the final-phase RTL small. Both use the custom-0 opcode (`0x0B`):
`qmac8` funct3 = 0, `rqnt` funct3 = 1. They are emitted from C with `.insn r` inline assembly. Every wrapper has a
pure-C fallback with identical semantics, and the fallbacks are tested against the same golden vectors.

### `qmac8`: 8-lane INT8 dot product (targets bottleneck 1)
`rd = sext( Σ_{i=0..7} int8(rs1.byte[i]) × int8(rs2.byte[i]) )`

One 64-bit load fetches 8 weights, one fetches 8 activations, and one instruction does 8 MACs. The sum is at most
8 × 16,384 in magnitude, so it is exact with no saturation.

- **Accumulating vs non-accumulating.** An accumulating form (`rd += dot8`) saves one `add` per 8 MACs, but it reads
  `rd` as a third source register, which the register file does not support. We chose the non-accumulating form;
  its cost is 1 instruction per 8 MACs.
- **Alignment.** 16 of the 32 1×1-conv weight buffers sat at a 4-byte offset inside the model file. 64-bit loads
  need 8-byte alignment (misaligned loads may not be supported by CV-Wally). We re-pack the `.tflite` file offline
  so every buffer is 16-byte aligned:
  - the weights are byte-identical
  - the interpreter output is verified identical
  - all configurations use the re-packed file
  - it costs nothing at runtime, unlike runtime copies (+1.66 MB) or two loads plus shifts per word

  B0–B1c instruction counts were unchanged by the re-pack.

### `rqnt`: one-instruction requantization (targets bottleneck 2)
Every output value must be scaled from the int32 accumulator back to int8 with TFLM's fixed-point
`MultiplyByQuantizedMultiplier`, then the output zero point is added and the result clamped (ReLU6 is folded into
the clamp). In scalar RV64IMC this takes about 52 instructions per output. `rqnt` does it in one:

- `rs1` = int32 accumulator (bias already added)
- `rs2` = all parameters packed into 64 bits: multiplier [31:0], shift [39:32], output zero point [47:40],
  clamp min [55:48], clamp max [63:56]
- `rd` = int8 result, bit-exact with TFLM's double rounding

Decisions:
- **Parameters in rs2 vs alternatives.** Splitting the work into two instructions (scale, then zero point + clamp)
  doubles the count. Keeping the parameters in CSRs would add architectural state. Packing everything into one
  64-bit word per output channel (built once at setup, 168 KB in total for the model) costs one load per output.
- **Bias add stays a separate `add`.** Adding the bias inside `rqnt` would need a third source register. The cost is
  one instruction per output.
- **Where it is used.** 1×1 conv, FC and depthwise. After `qmac8`, depthwise became the largest cost, and it
  requantizes after only 9 or 25 MACs per output, so it benefits most (Amdahl's law).
- **Reserved encodings.** Shift values outside −31…30 are reserved, which keeps the hardware shifter bounded. All
  layers of this model fall inside the range. The kernels check every layer's shift at setup and fall back to the
  scalar path if one does not.

### Ideas not used for the midterm
The course allows at most two instructions at midterm. The following were considered and deferred, because each
needs new state or a new datapath: fused load-MAC, post-increment loads, hardware loops, and an accumulator
register/CSR. These are candidates for the final-phase extension.

## 5. Verification strategy

| Level | Check |
|---|---|
| Instruction | Python golden model per instruction; directed edge cases (−128×−128, sum extremes, rounding ties, shift limits, saturation) plus 10,000 fixed-seed random vectors, run on Spike; the C fallback runs on the same vectors |
| ISA isolation | Each custom instruction traps as illegal on plain `--isa=rv64imc` |
| Kernel | Each optimized kernel is bit-exact against the TFLM reference kernel on every real layer shape of the model |
| Full model | All 1,000 output bytes are bit-exact against LiteRT `BUILTIN_REF` on three fixed inputs, for every configuration |
| No FP | An objdump scan of every ELF finds no F/D/A/V instructions |
