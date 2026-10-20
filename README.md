# Quantized Neural-Network Acceleration on RV64IMC (CV-Wally)

Architectural extensions for accelerating INT8 (W8A8) neural-network inference on a bare-metal
RV64IMC RISC-V processor, evaluated at the ISA level on Spike and at the RTL level on CV-Wally.

**Workload:** EfficientNet-Lite0, INT8 (TFLite Micro, bare-metal).
**Extensions:** packed INT8 multiply-accumulate (`qmac8`) and a requantization instruction (`rqnt`).

> Status: work in progress (midterm: ISA design and Spike evaluation).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Bare-metal runtime, inference kernels, Spike extension, tests |
| `scripts/` | Toolchain setup, model download/conversion, build, run, analysis |
| `experiments/` | Configurations B0, B1, X1, X2 |
| `results/` | Generated measurements and plots |
| `docs/` | ISA specification and figures |
| `DESIGN.md` | Design decisions and rationale |
| `EXPERIMENTS.md` | Methodology and results |
| `AI_USAGE.md` | Use of AI tools in this project |

## Quick start

_To be added._
