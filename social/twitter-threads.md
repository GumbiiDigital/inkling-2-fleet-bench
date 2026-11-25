# Inkling-2-Mini — X / Twitter Thread Series (Adversarially Audited)

Every number below is verified directly against raw `results/*.json` records on disk, and every thread includes a preemptive **"What to call out / Caveats"** block so technical readers see the exact limits upfront.

> **Tip for uploading charts to X/Twitter:** X does not accept `.svg` uploads directly—export 2× PNG screenshots from `charts/*-dark.svg` (or ask to batch-render them to PNG) before posting.

---

## Thread 1: Flagship Launch Thread (4 NVIDIA Blackwell Machines × 4 Engines)

### Tweet 1 (Hook + Chart)
**Attach:** `charts/01-single-stream-decode-dark.svg` (rendered as PNG)

We benchmarked Thinking Machines' **Inkling-2-Mini** (26.2B / 3.4B active MoE, looped layers, 1M context) across **4 NVIDIA Blackwell machines** and **4 engines** (`llama.cpp`, `vLLM`, `SGLang`, `Strata`):

- DGX Spark (GB10, 121 GB)
- RTX Spark (N1X, 128 GB)
- RTX PRO 6000 Blackwell (96 GB)
- RTX 5090 (32 GB)

Every point = median of 3 runs, adversarially audited against raw JSONs. Here's what held up—and the gotchas we caught 🧵👇

### Tweet 2 (Unified-Memory Sparks: Engine > Hardware)
**Attach:** `charts/03-concurrency-scaling-dark.svg` (rendered as PNG)

1/ **On unified-memory Sparks, engine choice is massive**

Moving DGX Spark & RTX Spark from `llama.cpp Q4_K_XL` → native NVFP4 (`vLLM` / `SGLang`):
- 128K prefill latency drops **6–7×** (364–385s → **51–63s**)
- 64-stream aggregate output jumps **2.3–2.5×**:
  - RTX Spark `SGLang`: **922.9 tok/s** (vs 374.5 `llama.cpp`)
  - DGX Spark `SGLang`: **842.0 tok/s** (vs 368.8 `llama.cpp`)

*(Important: we did NOT run vLLM/SGLang on PRO 6000 or 5090. On identical `llama.cpp Q4_K_XL`, PRO 6000 is still **2.42× faster** than RTX Spark: 906.8 vs 374.5 tok/s.)*

### Tweet 3 (DSpark Speculative Decoding)
**Attach:** `charts/02-dspark-reversal-dark.svg` (rendered as PNG)

2/ **DSpark speculative decoding splits sharply by engine**

- `llama.cpp` (early Unsloth `b12588-mix` GGUF drafter): **0.45×–0.58× slower** everywhere (6–14% acceptance)
- `Strata` native bench (`BEAM-W4 TREE-6` on PRO 6000): real, bit-exact **2.04× speedup** (**262.9 vs 128.7 tok/s**, 256/256 parity)—though Strata v0.2.19's HTTP serve API didn't engage it
- `vLLM`/`SGLang` on Sparks: **3.1×–4.1×** (120–183.5 tok/s), *with a big caveat in the final tweet*

### Tweet 4 (1M-Context Diagnostic)
**Attach:** `charts/06-1m-breakthrough-dark.svg` (rendered as PNG)

3/ **The 1M-Context Diagnostic (0/3 → 3/3)**

At 1,047,040 tokens, standard chat failed **0/3** on every engine (`llama.cpp`, `vLLM`, `SGLang`). Why?

Inkling-2-Mini's chat template ignores `reasoning_effort=none` (~98.4% reasoning tokens) and loops until hitting the output cap.

Bypassing the loop with a 4-word **assistant prefill** (`continue_final_message`) on DGX Spark `vLLM max-bf16` hit **3/3 exact needles in 9 output tokens** (~25.5 min/run)—proving the 1M KV state retains the fact, even though free-form 1M reasoning still collapses.

### Tweet 5 (RTX PRO 6000 Engine Trade-Offs + 450W = 600W)
**Attach:** `charts/05-rtx6000-engines-dark.svg` (rendered as PNG)

4/ **RTX PRO 6000 (96 GB): Every engine has a sharp trade-off**

- `Strata SQ-20`: Fastest single-stream (**178.6 tok/s**) & batching (**1,046.6 tok/s @64**), but **OOMs at 267K** on the 1M prompt
- `Strata #6520 NVFP4`: Fast 1-stream (**163.1**), best small-eval score (**0.867**), but **doesn't batch** (flat 140 tok/s)
- `Strata #6642 GGUF`: Trails `llama.cpp` on the exact same `IQ4_NL` file and early-EOSes
- `llama.cpp`: Most reliable, completes 1M, and **450 W matches 600 W** within 0.6% (serial soak draws ~221 W mean)

### Tweet 6 (Preemptive Caveats — "Before You Call Us Out")

5/ **Caveats & limitations (caught in our own raw-JSON audit):**

- **No vLLM/SGLang on PRO 6000 or 5090** — don't compare Spark `SGLang` to PRO 6000 `llama.cpp` as a hardware win
- **Track E @64** used an identical 512-char prompt with prefix/radix caching enabled (warm median)
- **Track J `vLLM`/`SGLang` 3.1×–4.1× DSpark** looped on 1-token NATO filler (`"mike mike..."`) → synthetic upper bound with wide variance (68–208 tok/s), NOT real-text throughput
- **1M prefill proof** is n=3, depth 0.5, single synthetic haystack on `vLLM max-bf16` — proves KV retention, not general 1M reasoning
- **Math/Code evals** (n=20/10) used default temp (not temp 0); Vision (n=8 synthetic squares) is a wiring smoke check only
- **RTX Spark H2H** ran in WSL2 (no power/temp telemetry, SGLang `mem-fraction 0.95` vs DGX `0.75`)

### Tweet 7 (Repo CTA + Apple Silicon v2.0 Teaser)

Full open repo with:
- Interactive dark/light HTML dashboard
- Normalized JSON & CSV datasets
- Deterministic dark/light SVG charts (matching our `omlx-fleet-bench` style)
- Full verify-PASS reports + raw audit caveats

🔗 [REPO LINK]

Next up (v2.0): **M3 Ultra 256GB** & **M4 Max 128GB** Metal benchmarks are finishing their 1M-context runs right now!

---

## Thread 2: Standalone Deep-Dive — The 1M-Context Diagnostic

### Tweet 1
**Attach:** `charts/06-1m-breakthrough-dark.svg` (rendered as PNG)

Testing Inkling-2-Mini at **1,047,040 tokens** gave **0/3** on `llama.cpp`, `vLLM`, and `SGLang`.

Before blaming the KV cache, we audited the traces and ran a controlled diagnostic on DGX Spark (`vLLM max-bf16`). The model *can* retrieve at 1M—the chat template was sabotaging it. 🧵

### Tweet 2
Even with `reasoning_effort: "none"`, Inkling-2-Mini's template still spends **~98.4% of its tokens** in `reasoning_content`.

At 1M tokens, it actually spotted the needle inside reasoning on several runs—then looped until `finish_reason=length` with an empty final answer.

### Tweet 3
Bypassing the reasoning loop with a 4-word **assistant prefill** (`"The hidden code is"`, `continue_final_message: true`) on DGX Spark `vLLM max-bf16`:

- Run 0 (`1,033,963` tok): `XJ9-4782.` — **9 tokens, clean stop**
- Run 1 (`1,046,579` tok): `XJ9-6498.` — **9 tokens, clean stop**
- Run 2 (`1,046,579` tok): `XJ9-7378.` — **9 tokens, clean stop**

**3/3 (100%)**, ~25.5 min/run.

### Tweet 4 (Preemptive Caveats + Repo Link)
**What this proves (and what it doesn't):**
- Proves the 1M KV attention state retains the needle (ruling out KV truncation/corruption)
- Does **NOT** prove general-purpose 1M reasoning: tested at depth 0.5 only, n=3, synthetic NATO haystack, single engine (`vLLM max-bf16`), and raw forced-prefix Arm B hallucinated `3X7-5778`

Full data & scripts: 🔗 [REPO LINK]
