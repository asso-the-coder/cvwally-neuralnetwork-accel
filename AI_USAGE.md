# AI Usage

AI coding assistants were used in this project as development tools.

**What the author did.** The author defined the project scope, workload and methodology, and made every design
decision: the choice of model and runtime, the baseline policy, each software optimization, the custom-instruction
constraints, semantics and encodings, and which kernels use them. The alternatives considered and the reasons for
each choice are documented in [DESIGN.md](DESIGN.md).

**What the AI tools did.**
- Helped analyze profiles, discuss design alternatives, and review the author's reasoning.
- Wrote much of the implementation from written task specifications: build scripts, bare-metal runtime glue,
  kernels, the Spike extension, test harnesses and analysis scripts.
- Drafted parts of the documentation.

**How correctness was ensured.**
- All AI-written code was reviewed against its specification and accepted only after automated tests passed:
  - Python golden models
  - directed and randomized instruction tests
  - kernel tests bit-exact against TFLite Micro reference kernels
  - full-model bit-exact checks against the LiteRT reference interpreter
- Tests were never weakened to make code pass.
- All reported numbers are produced by the scripts in `scripts/` and are reproducible from a clean clone.
