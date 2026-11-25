# Inkling 2 Mini — DGX Spark CUDA Benchmark Report

**Status: FINAL — verify gate PASS** (`python3 harness/verify.py results/` → PASS; 41 records, all approved tracks A/B/C/D/E/G/H/J/K measured, F blocked-informational). All tracks complete on DGX Spark 1 + Spark 2.

Native CUDA benchmark of Thinking Machines' **Inkling-2-Mini** on **NVIDIA DGX Spark** (GB10, sm_121, CUDA 13.0, 121 GB unified memory), driven from a reusable Python harness. Inspired by Pratik Gandhi's Mac benchmark; this report measures the Spark on its own terms — no cross-platform comparison is claimed.

## Setup

| Item | Value |
|---|---|
| Model | Inkling-2-Mini — 26.2B total / 3.4B active MoE, middle 12 of 24 layers looped 3×, 1M trained context, vision-capable |
| Hosts | spark-1 (192.168.72.232), spark-2 (192.168.72.54) — identical DGX Spark GB10 |
| Runtime A | Unsloth prebuilt llama.cpp **b12588** (CUDA, sm_121) — loads `inkling2` arch GGUFs (Q4_K_XL, Q8_K_XL, DSpark drafter, mmproj) |
| Runtime B | Mainline llama.cpp **418fe97** built on-box (CUDA 13.0, sm_121) — loads `inkling2_loop` arch GGUFs (BF16, IQ4_NL) |
| Quants | unsloth/Inkling-2-Mini-GGUF @ ea0c79b (Q4_K_XL, Q8_K_XL); ggml-org/Inkling-2-Mini-GGUF @ db61c5b (BF16, IQ4_NL) |
| Method | median of 3 runs per point; llama-server `/completion` streaming timings (`predicted_n/predicted_ms` authoritative for decode); per-point server lifecycle; evidence logs per record |
| Harness | `python3 -m harness.run --track <X>`; finish gate `python3 harness/verify.py results/` |

**Architecture finding:** the two GGUF publishers use different architecture tags. Unsloth files (`inkling2`) load only on the Unsloth build; ggml-org BF16/IQ4_NL (`inkling2_loop`) load only on mainline. The harness selects the runtime per quant automatically. Unsloth publishes no BF16 (Q8_K_XL is its top rung).

## Track A — Speed ladder (Q4_K_XL, spark-1)

Decode tok/s, median of 3, thinking on/off via `chat_template_kwargs.reasoning_effort`:

| Context | Thinking off | Thinking on | Prefill tok/s | TTFT (s) |
|---:|---:|---:|---:|---:|
| 512 | 38.53 | 37.46 | ~432 | 0.33 |
| 8,192 | 32.86 | 32.84 | ~547 | 14.3 |
| 32,768 | 28.74 | 28.45 | ~397 | 81.5 |
| 131,072 | 19.59 | 19.49 | ~359 | 363.6 |

**Finding — the thinking toggle does not reduce reasoning.** `reasoning_share ≈ 0.984` at every point, on *and* off: ~98% of generated tokens go to `reasoning_content` even at effort "none". At the 128-token benchmark budget, content tokens = 0 (the answer never starts). The toggle changes effort labeling, not the model's behavior at this budget. Longer-budget evals (Track C) do produce answers.

## Track B — Precision ladder

Decode tok/s, median of 3 (5-item math accuracy is a small-sample signal, not a full eval):

| Quant | Runtime | Host | @512 | @32k | Prefill @512 | Math (5-item) |
|---|---|---|---:|---:|---:|---:|
| Q4_K_XL | unsloth b12588 | spark-1 | 38.2 | 28.9 | 431 | 0.8 |
| Q4_K_XL | unsloth b12588 | spark-2 | 37.3 | 28.4 | — | 0.6 |
| IQ4_NL | mainline 418fe97 | spark-2 | 36.9 | 28.0 | — | 0.4–0.6 |
| Q8_K_XL | unsloth b12588 | spark-2 | 29.9 | 23.6 | — | 0.8 |
| BF16 | mainline 418fe97 | spark-1 | 19.9 | 16.7 | 520 | 0.6 |

- Q4_K_XL is the speed/quality sweet spot: ~2× BF16 decode at equal-or-better small-sample math.
- **BF16 prefill crossover:** BF16 prefills *faster* at 512 (520 vs 431 tok/s — compute-bound favors BF16 tensor cores) but *slower* at 32k (221 vs 403 — memory-bound favors quantized weights).
- Cross-host Q4 agreement (38.2 vs 37.3 @512) confirms the two Sparks are equivalent.

## Track C — Evals (Q4_K_XL, spark-1, thinking off)

| Suite | Score |
|---|---|
| Math (20 items) | **15/20 (0.75)** |
| Code (10 items) | **10/10 (1.0)** |
| Combined | 0.833 |

Budgets: math 1024, code 2048 tokens. (Reasoning-token metadata is unavailable on the non-benchmark chat path; the ~98% reasoning share finding comes from Track A's streaming path.)

## Track G — Vision (Q4_K_XL + mmproj, spark-1)

| Suite | Score | Determinism | Misses |
|---|---|---|---|
| Vision (8 items) | **5/8 (0.625)** | identical across 3 runs, zero truncated answers | 3 shape miscounts (8 shapes → answers 7, 7, 6) |

The misses are a genuine counting limitation, audited per item in `results/evidence/G-items-*.log`.

An earlier run scored 1/8; root cause was harness-side (default template effort + 512-token cap starved the answer). Fixed with thinking off, temperature 0, 1024 budget, and per-item audit logging.

## Track H — Soak / power (Q4_K_XL, spark-1, continuous load)

3 × 10-minute windows, back-to-back requests (~283/window):

| Window | Decode tok/s | GPU power mean | Temp mean | Temp max |
|---|---:|---:|---:|---:|
| 0 | 30.2 | 37.6 W | 62.4 °C | 64 °C |
| 1 | 30.2 | 37.4 W | 62.3 °C | 64 °C |
| 2 | 29.9 | 37.2 W | 61.9 °C | 64 °C |

**Flat throughput — no thermal throttling over 30 minutes.** ~0.81 tok/s per watt. (An earlier soak design slept between bursts — ~17% duty cycle — and understated load; replaced with continuous generation.)

## Track J — DSpark speculative decoding (Q4_K_XL + drafter, spark-1)

| Context | Baseline decode | Speculative decode | Acceptance |
|---:|---:|---:|---:|
| 512 | 38.97 | 17.95 | 8.3% |
| 32,768 | 29.26 | 15.81 | 13.5% |

**Negative result: speculation is ~2× slower.** The drafter's acceptance rate (8–13%) is far below the break-even point; draft+verify overhead dominates. Reported as measured — DSpark does not help this model on this hardware at these settings.

## Track F — Loop depth: BLOCKED (recorded, not failed)

The loop count is baked into the GGUF (`LLM_KV_LOOP_COUNT`); the shipped files cannot vary it at load time. Re-exported GGUFs with custom loop counts are required. Recorded as a blocked point with reason; verify treats it as informational.

## Track D — Needle in a haystack (Q4_K_XL, spark-2)

One fact per document, chat grading on the code token in content, temperature 0, 2048-token budget, needle never trimmed. Runs on spark-2 (identical hardware/model/runtime; no interference with lane 1).

| Context | Depth | Content accuracy (median) | Fact retrieved¹ | Clean stops |
|---:|---:|---:|---:|---:|
| 16,384 | 0.1 / 0.5 / 0.9 | 1.0 / 1.0 / 1.0 | 3/3 each | 3/3 each |
| 65,536 | 0.1 / 0.5 / 0.9 | 1.0 / 1.0 / 1.0 | 3/3 each | 3/3 each |
| 131,072 | 0.5 | 1.0 | 3/3 | 3/3 |
| 262,144 | 0.5 | 0.0 | 2/3 | 1/3 |
| 1,047,040 | 0.5 | 0.0 | 0/3 | 0/3 |

¹ "Fact retrieved" = the needle code appears in the answer **or** in the reasoning trace (secondary metric). "Clean stops" = finished within the 2048-token answer budget without truncation.

**7/7 points perfect through 128k** (all depths, zero budget hits). At **256k** the fact is still retrieved (2/3 runs — one clean answer, two found in reasoning), but reasoning length grows and exhausts the 2048-token answer budget before the answer is emitted, so content accuracy drops. At **1M** the full 1,048,576-token slot loads and runs (~59 GB of 121 GB), but within a 2048-token budget the reasoning consumes the cap without surfacing the fact (0/3, all `finish_reason=length`).

**1M retrieval probe (supplementary, definitive):** a single-run probe with an **8192-token** budget (4× standard; identical document/seed, temperature 0) on spark-1 also failed — `needle_accuracy 0.0`, `found_in_reasoning false`, full 8192 tokens consumed (`finish_reason=length`), 19,403 chars of reasoning, empty answer. Inspection of the reasoning trace shows **degenerate repetition**: the model loops over filler words ("mike, oscar, india, juliet, quebec…") instead of searching for the fact. So the 1M failure is **not** an answer-budget artifact — coherent retrieval breaks down at 1M context (0/3 standard + 0/1 probe). Evidence: `results/evidence/D-1M-probe-8192-spark-1.{json,log}`.

**Headline:** 1M context **loads and runs on one Spark** (full 1,048,576 slot, ~59 GB), but **retrieval fails at 1M** (degenerate generation). Retrieval is perfect through 128k, partial at 256k (fact found in reasoning, answer budget exhausted), and breaks down at 1M. Decode also slows to ~4 tok/s at 1M (vs ~20 at 128k) as attention spans the full cache.

## Track E — Concurrency (Q4_K_XL, spark-2, on-box load)

Aggregate output tok/s, median of 3, **zero failed workers at every level**:

| Parallel | Output tok/s | p50 latency | p95 latency | Server ctx |
|---:|---:|---:|---:|---:|
| 1 | 36.8 | 7.0 s | 7.0 s | 8,192 |
| 4 | 86.1 | 11.9 s | 11.9 s | 8,192 |
| 16 | 196.2 | 20.9 s | 20.9 s | 8,192 |
| 64 | 368.8 | 44.4 s | 44.4 s | 32,320 |

Throughput scales ~10× from 1→64 concurrent requests (36.8 → 368.8 tok/s); per-request latency rises as slots share the GPU. Load is generated on-box (one SSH session, local HTTP threads).

An earlier attempt fanned out one SSH session per worker from the Mac; the Sparks' hardened sshd (MaxSessions 4, MaxStartups 10:30:30) dropped connections beyond ~10 (3/16 and 24/64 workers failed), understating throughput (34.9/70.3/112.6/180.4). Those records are preserved in `results/archive-ssh-fanout/`. A server-restart port-release race (brief 22) was also fixed before the clean re-run.

## Track K — Two-node over 200G fabric (Q4_K_XL, RPC offload)

llama-server on spark-1 with `--rpc 10.230.0.2:50052` (ggml-rpc-server on spark-2), fabric link (0% loss, ~1.4 ms RTT; both rails 200 Gbps, UP). Median of 3:

| Context | Decode tok/s | Prefill tok/s | TTFT (s) |
|---:|---:|---:|---:|
| 512 | 5.83 | 128.8 | 0.99 |
| 32,768 | 6.41 | 259.4 | 124.8 |

**Two-node inference works over the fabric**, but it is **slower than single-box** for this model (single-box Q4: 38.5 / 28.7 tok/s decode). Splitting a model that already fits on one Spark across two nodes adds cross-node RPC round-trips per token, so throughput drops ~6×. RPC offload's value is running models **too large for one box's memory**, not accelerating ones that fit. Reported as measured.

## Harness engineering notes

- 23 writer briefs; 146 unit tests; verify gate (≥3 runs, medians, evidence per approved track×host; blocked informational; failed = FAIL) → **PASS**, 41 records.
- Defects found and fixed live: arch mismatch (mainline build), needle grading (chat content vs completion), reasoning starvation at small budgets (temperature 0 + larger caps + audit fields), ctx-scaled request timeouts (256k prefill > 600 s), soak duty cycle, vision determinism, SSH fan-out concurrency (on-box load), server stop port-release race, K rpc-server graceful cleanup.
- All superseded/defective records preserved in `results/archive-*` (verify reads top-level only).

## Reproduce

```bash
python3 -m harness.run --track A        # speed ladder
python3 -m harness.run --track B --host spark-1
python3 -m harness.run --track D --host spark-2
python3 harness/verify.py results/      # finish gate
```

Records: `results/<track>-<host>-<params>.json`; evidence: `results/evidence/`.