# Inkling-2-Mini — X / Twitter Thread Series

Ready-to-post threads with exact chart attachments from `charts/`. Every figure is verified against `data/benchmarks.json` and the PASS-gated reports in `reports/`.

---

## Thread 1: Flagship Launch Thread (4 NVIDIA Blackwell Machines × 4 Engines)

### Tweet 1 (Hook + Chart)
**Attach:** `charts/01_single_stream_decode.png`

We just benchmarked Thinking Machines' **Inkling-2-Mini** (26.2B / 3.4B active MoE, looped layers, 1M context) across **4 NVIDIA Blackwell machines** and **4 inference engines** (`llama.cpp`, `vLLM`, `SGLang`, `Strata`).

- DGX Spark (GB10)
- RTX Spark 128GB (N1X)
- RTX PRO 6000 Blackwell (96GB)
- RTX 5090 (32GB)

Every point = median of 3 runs, verify-gated. Here's what we actually found 🧵👇

### Tweet 2 (The DSpark Reversal)
**Attach:** `charts/02_dspark_reversal.png`

1/ **The DSpark Speculative Decoding Reversal**

On `llama.cpp`, DSpark makes Inkling-2-Mini **~2× SLOWER** on every single box (0.45×–0.58×, 6–13% acceptance).

Switch to native `vLLM` or `SGLang` and it flips into a **3.1×–4.1× SPEEDUP**:
- RTX Spark `vLLM`: **183.5 tok/s** (4.1×!)
- DGX Spark `vLLM`: **142.0 tok/s** (3.8×)
- RTX PRO 6000 `Strata` native: **262.9 tok/s** (2.04×, bit-exact)

Same model. Completely opposite outcome depending on the engine.

### Tweet 3 (Concurrency Scaling)
**Attach:** `charts/03_concurrency_scaling.png`

2/ **Concurrency Scaling (1 → 64 Parallel Streams, 0 Failures)**

At 64 concurrent requests:
- RTX PRO 6000 `Strata SQ-20`: **1,046.6 tok/s**
- RTX PRO 6000 `llama.cpp IQ4_NL`: **978.0 tok/s**
- RTX Spark `SGLang`: **922.9 tok/s** 🤯
- DGX Spark `SGLang`: **842.0 tok/s**
- RTX 5090 `llama.cpp`: **601.7 tok/s**

Wild takeaway: `SGLang` on the 128GB RTX Spark laptop matches a single RTX PRO 6000 running `llama.cpp Q4_K_XL` (907 tok/s) at 64 streams!

### Tweet 4 (The 1M-Context Breakthrough)
**Attach:** `charts/06_1m_context_breakthrough.png`

3/ **The 1M-Context Plot Twist (0/3 → 3/3)**

At 1,047,040 tokens, *every* engine (`llama.cpp`, `vLLM`, `SGLang`) scored **0/3** on standard chat. Why?

Inkling-2-Mini's chat template ignores `reasoning_effort=none` (~98.4% reasoning tokens), so it loops until hitting the output cap.

Bypass the reasoning loop with **assistant prefill (`continue_final_message`)** on DGX Spark `vLLM max-bf16` → **3/3 (100%) exact needle recall in 9 output tokens.**

### Tweet 5 (RTX PRO 6000 Power Paradox)
**Attach:** `charts/04_rtx6000_power_knee.png`

4/ **The RTX PRO 6000 Power-Cap Paradox (350W vs 450W vs 600W)**

Raising the power cap from 350 W → 450 W:
- Decode @128K: **+167%** (24.0 → 63.9 tok/s)
- 1-stream concurrency: **+221%** (28.8 → 92.8 tok/s)
- 30m soak power: **DROPPED from 327 W → 221 W**
- Max temp: **DROPPED from 84 °C → 66 °C**

450 W and 600 W are identical. Escaping the 350 W SW Power Cap lets the GPU boost, finish faster, and run cooler.

### Tweet 6 (Strata Engine Trade-Offs)
**Attach:** `charts/05_rtx6000_engines_and_quants.png`

5/ **Strata vs llama.cpp on RTX PRO 6000 (600W)**

Every engine has a sharp trade-off:
- `Strata SQ-20`: Fastest single-stream (**178.6 tok/s**) & batching (**1,047 tok/s**), but OOMs at 1M (267K prefill)
- `Strata #6520 NVFP4`: Best eval score (**0.867**) & fast 1-stream (**163.1**), but **doesn't batch** (flat 140 tok/s)
- `Strata #6642 GGUF`: Trails `llama.cpp` on the exact same `IQ4_NL` file
- `llama.cpp`: Most reliable, runs full 1M

### Tweet 7 (Repo CTA + Apple Silicon Teaser)

Full open repo with:
- Interactive HTML benchmark dashboard
- Normalized JSON & CSV datasets
- All 6 charts + reproducible scripts
- Full verify-PASS reports for every box

🔗 [REPO LINK]

Next up (v2.0): **M3 Ultra 256GB** & **M4 Max 128GB** Metal benchmarks are finishing their 1M-context runs right now—stay tuned!

---

## Thread 2: Standalone Deep-Dive — The 1M-Context Fix (High-Engagement Technical Thread)

### Tweet 1
**Attach:** `charts/06_1m_context_breakthrough.png`

Everyone testing Inkling-2-Mini at 1M context is hitting 0/3 retrieval and blaming the model or KV cache.

We tested `llama.cpp`, `vLLM`, and `SGLang` at **1,047,040 tokens**—and found the real culprit isn't retrieval. It's the chat template. 🧵

### Tweet 2
Even when you pass `reasoning_effort: "none"`, Inkling-2-Mini still spends **~98.4% of its output tokens** inside `reasoning_content`.

At 1M tokens, it actually *finds* the needle in its reasoning trace on several runs—then gets trapped looping until `finish_reason=length` with an empty final answer.

### Tweet 3
So we tested a 1-line fix on DGX Spark (`vLLM max-bf16`, `1,047,040` ctx):

Use **assistant prefill** (`continue_final_message: true`) to bypass the reasoning loop completely.

Result across 3 independent seeds:
- Run 0 (`1,033,963` tok): `XJ9-4782.` — **9 tokens, clean stop**
- Run 1 (`1,046,579` tok): `XJ9-6498.` — **9 tokens, clean stop**
- Run 2 (`1,046,579` tok): `XJ9-7378.` — **9 tokens, clean stop**

**3/3 (100%) exact 1M recall.**

### Tweet 4
Full JSON receipts, reproduction scripts, and all 4 NVIDIA Blackwell machine reports are live in our benchmark repo:

🔗 [REPO LINK]

---

## Thread 3: Standalone Deep-Dive — RTX PRO 6000 Power-Cap Paradox

### Tweet 1
**Attach:** `charts/04_rtx6000_power_knee.png`

Counterintuitive GPU physics from our RTX PRO 6000 Blackwell benchmarks:

Raising the power limit from **350 W → 450 W** made Inkling-2-Mini **+167% to +221% faster**—while dropping average power by **106 W** and max temp by **18 °C**.

Here's why 🧵

### Tweet 2
At 350 W, the card was pinned against `SW Power Cap: Active` (SM clocks throttled to ~1800 MHz).

Because every request took much longer, the GPU stayed at a high duty cycle—averaging **327 W** at **84 °C** with a 15% throughput fade over 30 minutes.

### Tweet 3
At 450 W (and 600 W), the workload's natural peak draw is only **~230 W**.

With the cap out of the way, the card boosts to full clocks, finishes requests 67%–221% faster, and settles at **221 W mean / 66 °C max** with zero fade—**2.5× better tok/W** (0.299 vs 0.121).

450 W buys 100% of the 600 W performance. Full data in the repo: 🔗 [REPO LINK]
