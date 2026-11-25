# Inkling 2 Mini — Titan RTX PRO 6000 Blackwell CUDA Benchmark Report (600 W, with 350 / 450 / 600 W power ladder)

**Strata addendum:** a native-engine comparison (Strata v0.2.19, SQ-20, single card) was added on 2026-11-23 — see the last section. Strata is ~2× faster at short context and matches accuracy, but its engine cannot process the 1M needle point (CUDA OOM). Its verify gate also PASSes (`results-strata/`).

**Status: FINAL — verify gate PASS on all three runs.** 600 W: `python3 harness/verify.py results-600w/ --config harness/config.titan.600w.json` → PASS (38 records, 0 failed). 450 W: `python3 harness/verify.py results-450w/ --config harness/config.titan.450w.json` → PASS (38 records, 0 failed). 350 W baseline: PASS (38 records, 0 failed), preserved at `baseline-350w-accepted/`. Approved tracks A/B/C/D/E/G/H measured, J measured, F/K blocked-informational. All tracks complete on Titan.

Native CUDA benchmark of Thinking Machines' **Inkling-2-Mini** on **Titan** (NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB, sm_120), driven from the same reusable Python harness as the DGX Spark and M3 Ultra sibling jobs. This report measures Titan on its own terms — no controlled cross-platform comparison is claimed.

**This report covers a three-point power ladder: 350 / 450 / 600 W.** The first suite (2026-11-21) ran under a **pre-existing enforced 350 W cap** and is the accepted baseline, preserved intact at `baseline-350w-accepted/` (38 records + its report + suite log). On 2026-11-22 the owner authorized two re-runs at higher caps: **600 W** (the card default, `results-600w/`) and **450 W** (a middle point, `results-450w/`). **Same box, same runtimes, same weights, same harness/config — only the power limit changed.** The three suites ran on different days (not simultaneously), each point a median of 3 runs. This makes it a controlled *power-envelope* comparison on Titan itself. The per-track tables below show the 600 W run; **the 450 W run matches it within run-to-run noise at every point** (see the power-ladder section). The card's hardware power range is **150 W min, 600 W default = max** — 600 W is the ceiling; higher settings (e.g. 750 W) are not accepted by this card.

## Headline — what raising the cap changed

| Signal | 350 W (baseline) | 600 W (this run) | Change |
|---|---:|---:|---|
| A decode, ctx-512 thinking-off | 67.27 tok/s | 86.76 tok/s | **+29%** |
| A decode, ctx-131k thinking-on | 23.95 tok/s | 63.89 tok/s | **+167%** |
| E aggregate, 1 worker | 28.82 tok/s | 92.65 tok/s | **+221%** |
| E aggregate, 64 workers | 897.28 tok/s | 906.77 tok/s | +1% |
| H soak decode | 39.54 tok/s | 66.14 tok/s | **+67%** |
| H requests / 600 s window | 379→371→322 (fading) | 621/620/621 (stable) | **no fade** |
| H max temp | 84 °C | 66 °C | cooler |
| H efficiency | 0.121 tok/W | 0.299 tok/W | **+147%** |
| J speculation penalty | ~4× slower | ~1.6× slower | still negative, milder |
| C / G / D accuracy | 0.80 / 0.625 / 9-of-14 | 0.80 / 0.625 / 9-of-14 | **unchanged** |
| Suite wall-clock | 3 h 51 m | 2 h 33 m | faster |

**Reading:** the 350 W cap was the binding constraint. Under it the card throttled (SM clock 1800 MHz vs 3090 max, "SW Power Cap: Active"), which produced the erratic speed ladder, the hot-suite slowdown, and the sustained-load fade. At the 600 W default the card is **not cap-bound** (H soak averaged 221 W, max 230 W — well below 600 W), the ladder is smooth, and sustained throughput is stable and cooler. **Accuracy is unchanged** — power affects speed, not correctness. Full per-point deltas: `evidence/compare-350w-vs-600w-2026-11-22.txt`; 600 W detail: `evidence/detail-600w-2026-11-22.txt`.

## Power ladder — 350 / 450 / 600 W (the three-point comparison)

| Signal | 350 W | 450 W | 600 W |
|---|---:|---:|---:|
| A decode, ctx-512 thinking-off (tok/s) | 67.27 | 87.26 | 86.76 |
| A decode, ctx-32k thinking-off (tok/s) | 28.10 | 77.55 | 77.69 |
| A decode, ctx-131k thinking-on (tok/s) | 23.95 | 63.92 | 63.89 |
| E aggregate, 1 worker (tok/s) | 28.82 | 92.79 | 92.65 |
| E aggregate, 64 workers (tok/s) | 897.28 | 904.20 | 906.77 |
| H soak decode (tok/s) | 39.54 | 65.72 | 66.14 |
| H requests / 600 s window | 379→371→322 (fading) | 617/615/620 (stable) | 621/620/621 (stable) |
| H mean power (W) | 327 | 221 | 221 |
| H max temp (°C) | 84 | 67 | 66 |
| H efficiency (tok/W) | 0.121 | 0.299 | 0.299 |
| C combined eval | 0.80 | 0.767 | 0.80 |
| G vision accuracy | 0.625 | 0.625 | 0.625 |
| D needle: found-in-reasoning (failed runs) | 9/14 | 9/14 | 9/14 |
| Suite wall-clock | 3 h 51 m | 2 h 35 m | 2 h 33 m |

**Key finding — the knee is below 450 W; 450 W already buys the full speedup.** The 450 W and 600 W runs are **indistinguishable within run-to-run noise at every speed point** (A ctx-512: 87.26 vs 86.76; E 1-worker: 92.79 vs 92.65; H decode: 65.72 vs 66.14). Raising the cap from 450 W to 600 W changes nothing measurable. The entire gain comes from escaping the **350 W** cap: 350 → 450 already delivers +29% (A ctx-512), +167% (A ctx-131k thinking), +221% (E 1-worker), and removes the sustained-load fade.

**Why:** the H soak shows the mechanism directly. At **350 W** the card was *pinned against its cap* (mean 327 W, max 352–356 W ≈ the 350 W limit, SW Power Capping active, clocks throttled) — a throttled equilibrium with a high duty cycle, fading throughput (379→322 requests/window) and 84 °C. At **450 W and 600 W** the card is *not cap-bound*: the soak peaks at only ~231 W (450 W) / ~230 W (600 W), far below either cap, so the card boosts to full clocks, completes requests faster, and settles at a lower-average-power (221 W), stable, cooler (66–67 °C) equilibrium. The natural draw of this workload (~221 W mean, ~231 W peak in the soak) sits **below 450 W**, so any cap at or above ~450 W is already non-binding. We did not test between 350 and 450, so the exact knee is not pinpointed — only that **350 W binds and 450 W does not**. Finer steps (e.g. 375/400 W) would locate it; the owner did not request them.

**Accuracy is unchanged across all three** (power affects speed, not correctness): G vision 0.625 everywhere; D needle hit counts identical at every point and 9/14 failed runs found the code in reasoning at all three caps. C combined read 0.767 at 450 W vs 0.80 at 350/600 — a **single math item's** run-to-run sampling difference (math 13/20 vs 14/20; code 10/10 everywhere), not a power effect.

**J speculation is noisy but negative at every cap.** Acceptance stays below break-even (6–22% across runs); speculative decode is slower than no-speculation at 350, 450 and 600 W. The penalty is milder at higher caps (~4× at 350 W; ~1.8× at 450 W and ~1.7× at 600 W for ctx-512) but J's acceptance/decode vary run-to-run, so no fine power trend is claimed. DSpark does not help this model at these settings.

Full 3-way per-point deltas (38/38 matched): `evidence/compare-3way-350-450-600-2026-11-22.txt`. 450 W detail: `evidence/detail-450w-2026-11-22.txt`. 350→600 deltas: `evidence/compare-350w-vs-600w-2026-11-22.txt`. 600 W detail: `evidence/detail-600w-2026-11-22.txt`.

## Setup

| Item | Value |
|---|---|
| Model | Inkling-2-Mini — 26.2B total / 3.4B active MoE, middle 12 of 24 layers looped 3×, 1M trained context, vision-capable |
| Host | titan (10.230.0.20, fabric) — RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB, compute cap 12.0, driver 595.99.02, CUDA 12.8, 64 cores, 251 GB RAM |
| Runtime A | Unsloth llama.cpp **b12588-mix** (commit 883286076) built from source on-box for sm_120 — loads `inkling2` arch GGUFs (Q4_K_XL, Q8_K_XL, DSpark drafter, mmproj) |
| Runtime B | Mainline llama.cpp **418fe97** (build 12602, commit 418fe97b5) built on-box (CUDA 12.8, sm_120) — loads `inkling2_loop` arch GGUFs (BF16, IQ4_NL) |
| Quants | unsloth/Inkling-2-Mini-GGUF (Q4_K_XL 17.6G, Q8_K_XL 22.9G, drafter 0.4G, mmproj-F16 0.6G); ggml-org/Inkling-2-Mini-GGUF (BF16 52G, IQ4_NL 17.6G) — staged direct from spark-1/spark-2, all sha256 verified (`evidence/stage.log`: 7/7 OK). **Same weights as the 350 W baseline.** |
| Power envelope | **Three points: 350 / 450 / 600 W.** Card range 150 W min, 600 W default = max (live-verified; 600 W is the ceiling, higher not accepted). 350 W was a pre-existing enforced cap (the accepted baseline). 450 W and 600 W set on 2026-11-22 by owner authorization (`sudo nvidia-smi -pm 1; sudo nvidia-smi -pl <W>`; verified `power.limit`, persistence Enabled). 350 W readings: `baseline-350w-accepted/` + `evidence/power-cap-2026-11-21.txt`. |
| Method | median of 3 runs per point; llama-server `/completion` streaming timings (`predicted_n/predicted_ms` authoritative for decode); per-point server lifecycle; evidence logs per record |
| Harness | `python3 -m harness.run --track <X> --config harness/config.titan.600w.json`; finish gate `python3 harness/verify.py results-600w/ --config harness/config.titan.600w.json`. The 600 W config is identical to the Titan config except `results_dir`/`evidence_dir` point at `results-600w/` so the accepted 350 W `results/` is never touched. |

**Architecture finding (unchanged from baseline, live-verified):** mainline 418fe97 **cannot** load the unsloth `inkling2` GGUFs (`check_tensor_dims: blk.24.attn_q.weight not found`). The Spark split is mirrored: unsloth files on the b12588-mix runtime, ggml-org files on mainline. The exact mix source (commit 883286076) was recovered from Unsloth CI run 289 artifacts (zip sha256 verified against the GitHub digest) and built for sm_120; `--spec-type draft-dspark` present. See the baseline report for the full evidence trail (`evidence/smoke-v1.log`, `evidence/mix-source-sha256.txt`, `evidence/MIX-BUILD-DONE.json`).

**Smoke proof (before the baseline suite; runtimes unchanged for this re-run):** mix+Q4+mmproj, mix+Q4+dspark drafter, and mainline+IQ4_NL all healthy → "391" (`evidence/SMOKE2-DONE.json`, `evidence/smoke2.log`).

## Track A — Speed ladder (Q4_K_XL, mix runtime)

Decode tok/s, median of 3, thinking on/off via `chat_template_kwargs.reasoning_effort`:

| Context | Thinking off | Thinking on | Prefill tok/s (off) | TTFT s (off) |
|---:|---:|---:|---:|---:|
| 512 | 86.76 | 87.03 | 992 | 0.14 |
| 8,192 | 82.22 | 82.17 | 2162 | 3.62 |
| 32,768 | 77.69 | 77.51 | 1865 | 17.37 |
| 131,072 | 63.96 | 63.89 | 1679 | 77.86 |

**Finding — the ladder is now smooth and the thinking toggle barely matters.** Decode decays monotonically with context (86.8 → 82.2 → 77.7 → 64.0), and thinking-on vs off differ by <0.3 tok/s at every point. This is a sharp contrast with the 350 W baseline, where the ladder was non-monotonic (32k at 28.1 slower than 131k at 50.8) and thinking-on 131k collapsed to 23.95. The clean 600 W ladder supports the baseline's interpretation that those distortions were power-cap throttling, not model behavior. Per-point clock telemetry was not recorded for this run either, so the clock-level mechanism remains inference; the cap-bound vs not-cap-bound distinction is directly supported by the H power readings below.

**Finding — the thinking toggle still does not reduce reasoning.** `reasoning_share ≈ 0.984` at every point, on *and* off: ~98% of generated tokens go to `reasoning_content` even at effort "none". At the 128-token benchmark budget, content tokens = 0. Same as the baseline and both siblings.

## Track B — Precision ladder

Decode tok/s, median of 3 (5-item math accuracy is a small-sample signal, not a full eval):

| Quant | Runtime | @512 | @32k | Prefill @512 | Math @512 (5-item) | Math @32k |
|---|---|---:|---:|---:|---:|---:|
| Q4_K_XL | mix b12588 | 87.34 | 77.61 | 2171 | 0.6 | 0.8 |
| Q8_K_XL | mix b12588 | 81.41 | 72.70 | 2205 | 0.6 | 0.4 |
| IQ4_NL | mainline 418fe97 | 94.66 | 81.11 | 2392 | 0.6 | 0.8 |
| BF16 | mainline 418fe97 | 70.37 | 62.80 | 1694 | 0.8 | 0.6 |

**The hot-suite slowdown is gone.** At 350 W, B ran hot right after the H soak and Q4@512 measured 28.37 (vs a cold A reference of 67.27). At 600 W, B Q4@512 = 87.34 ≈ the A cold reference 86.76 — the cap no longer depresses hot points. The 5-item math accuracies vary between runs (e.g. Q4@512 0.6 here vs 0.8 at 350 W); with only 5 items this is sampling noise, not a power effect, and is reported as measured.

## Track C — Evals (Q4_K_XL, thinking off, 30 items)

| Metric | Score |
|---|---:|
| Math (20 items) | 0.70 |
| Code (10 items, sandboxed tests) | 1.00 |
| Combined | 0.80 |

**Identical to the 350 W baseline** (math 0.70, code 1.00, combined 0.80). Thinking off, no temperature override, budgets 1024 (math) / 2048 (code); 2639 solve tokens total (~88/item). Code perfect (10/10); math 14/20. Per-item answers/token counts were not recorded, so the miss cause is not diagnosed (same limitation as baseline). Power does not change correctness.

## Track G — Vision (Q4_K_XL + mmproj-F16, 8 items)

Accuracy **0.625** (5/8), zero empty answers, zero budget hits — **identical to baseline**. Deterministic settings: thinking off, temperature 0, 1024-token budget.

## Track H — Soak / power (Q4_K_XL, 3 × 600 s continuous load)

| Window | Requests | Power mean W | Power max W | Temp mean °C | Temp max °C |
|---:|---:|---:|---:|---:|---:|
| 0 | 621 | 221.1 | 227.5 | 64.4 | 66 |
| 1 | 620 | 222.2 | 229.7 | 64.7 | 66 |
| 2 | 621 | 221.5 | 230.0 | 64.9 | 66 |

Median decode **66.14 tok/s**; **0.299 tok/W**. No crashes across **1862 requests**; every request completed (a failed request raises and aborts the window; all three finished).

**Finding — no sustained-load fade at 600 W; the 350 W fade was a cap artifact.** Requests/window are stable (621/620/621) versus the baseline's 379→371→322 (~15% fade). The card runs cooler (66 °C max vs 84 °C) and is **not cap-bound**: mean power 221 W / max 230 W sits well below the 600 W limit, whereas the 350 W soak was pinned against its cap (mean 320–337 W, max 352–356 W ≈ the 350 W limit, SW Power Capping active). Throughput is 67% higher and efficiency 2.5× better (0.299 vs 0.121 tok/W). The lower *absolute* average power at 600 W reflects the request-driven workload completing faster (a different duty cycle), not a measurement artifact — both runs used the same nvidia-smi sampling. Per-point clocks were not logged, so the boost-clock mechanism is inference; the cap-bound (350 W) vs not-cap-bound (600 W) distinction is measured.

## Track J — DSpark speculative decoding (Q4_K_XL + drafter, mix runtime)

| Context | No-spec decode (600 W) | Speculative decode | Acceptance |
|---:|---:|---:|---:|
| 512 | 87.13 | 50.21 | 0.06 |
| 32,768 | 77.22 | 49.36 | 0.14 |

**Still a negative result, but milder.** Speculation remains slower than no-speculation — ~1.7× at 512 and ~1.6× at 32k (versus ~4× at 350 W). Acceptance 6–14% is still below break-even, so draft+verify overhead still dominates; DSpark does not help this model at these settings. Raising the cap lifted both baseline and speculative decode, narrowing the gap, but did not flip the sign. This reproduces the Spark negative finding on different hardware with the same source-built runtime. J ran for real (mix build carries `--spec-type draft-dspark`).

## Track D — Needle in a haystack (Q4_K_XL, full plan to 1M)

Hits out of 3 runs per point (standard 2048-token answer budget, temperature 0):

| Context | depth 0.1 | depth 0.5 | depth 0.9 | Fact found in reasoning (failed runs) |
|---:|---:|---:|---:|---|
| 16,384 | 3/3 | 3/3 | 2/3 | 1/1 |
| 65,536 | 3/3 | 1/3 | 1/3 | 3/4 |
| 131,072 | — | 0/3 | — | 2/3 |
| 262,144 | — | 0/3 | — | 2/3 |
| 1,047,040 (1M) | — | 0/3 | — | 1/3 |

**Identical to the 350 W baseline** — every point's hit count matches, and again **9 of 14 failed runs found the hidden code in the reasoning trace** but exhausted the budget before emitting an answer (`finish_reason=length`). The full 1,048,576-token slot loads and runs on one 96 GB card. The two-stage failure (budget exhaustion first, then answer-emission breakdown beyond ~65k) is a model/budget capability limit, **not** a power effect — raising the cap does not change retrieval. The baseline's 8192-budget probes (both failed, `found_in_reasoning=true`) remain the reference for the ≥131k answer-emission breakdown; they were not re-run at 600 W because D's standard-budget result is unchanged.

**Headline:** usable needle answers stop at ~16k (all depths) / 65k shallow, exactly as at 350 W. Power buys speed, not long-context retrieval.

## Track E — Concurrency (Q4_K_XL, on-box load, 256-token answers)

| Parallel | Aggregate output tok/s | p50 s | p95 s | Failed workers |
|---:|---:|---:|---:|---:|
| 1 | 92.65 | 2.76 | 2.76 | 0 |
| 4 | 228.28 | 4.49 | 4.49 | 0 |
| 16 | 519.81 | 7.88 | 7.88 | 0 |
| 64 | 906.77 | 18.05 | 18.06 | 0 |

Zero failures at every level; p95 within 0.010 s of p50 at 64. **Low concurrency gains the most:** 1 worker +221% and 4 workers +172% versus 350 W, because single/few-stream decode is clock-bound and the cap had been throttling it. **High concurrency barely moves** (16 workers +0.9%, 64 workers +1.1%) — at 64 streams the aggregate was already bandwidth/batch-bound, not cap-bound, at 350 W. The 64-vs-1 aggregate ratio is now ~9.8× (it was ~31× at 350 W, where single-stream was heavily suppressed).

## Track F — Loop depth: BLOCKED (recorded, not failed)

Loop count is baked into the GGUF (`LLM_KV_LOOP_COUNT`); varying it requires re-exported GGUFs (`custom_loop_ggufs`), which do not exist. Blocked record written, same as baseline and both siblings.

## Track K — Two-node RPC: BLOCKED (single host)

Titan is one box; track K requires two distinct hosts. Blocked record written by the harness's single-host guard.

## Harness engineering notes

- **Host-agnostic adaptation (Cursor T1, Quick check SHIP):** per-host SSH target/identity options, single-host lane fallback, model paths resolved by host, precision matrix from config, J/K blocked guards, `verify --config`. Backward compatible: Spark reference config and all 146 original tests untouched; 19 new tests → **165 green**.
- **600 W re-run isolation:** a config variant (`config.titan.600w.json`) routes output to `results-600w/` so the accepted 350 W `results/` is never overwritten. This matters because the writer names records `{track}-{host}-{quant-params-slug}.json` with no timestamp — a same-config re-run would otherwise clobber the baseline. The accepted baseline is also snapshotted at `baseline-350w-accepted/`.
- **Runtime correction (baseline):** smoke v1 failed (mainline vs unsloth arch); corrected to the Spark split with the exact CI-artifact mix source built for sm_120; smoke v2 proved all three paths before any benchmark.
- Transport: SSH-exec + on-box python3 HTTP (no tunnels); server binds 127.0.0.1 on Titan.
- Record labeling: `runtime_version` is a static label (`unsloth-llama.cpp`) on every record; the actual runtime per point is `params.runtime` (track B) and the bin-path files.

## Reproduce

```bash
cd "Inkling 2 [RTX 6000]"
python3 -m unittest discover -s harness/tests -t .                 # 165 tests
# set the cap for the point you want (card range 150-600 W), e.g. 600 W:
sudo nvidia-smi -pm 1 && sudo nvidia-smi -pl 600
python3 -m harness.run --track A --config harness/config.titan.600w.json   # any track A-K
python3 harness/verify.py results-600w/ --config harness/config.titan.600w.json  # finish gate -> PASS
# 450 W point: sudo nvidia-smi -pl 450, then use config.titan.450w.json / results-450w/
```

Runtimes on Titan: `~/inkling-bench/runtime/mix-bin-path.txt` (mix b12588, sm_120), `~/inkling-bench/runtime/mainline-bin-path.txt` (418fe97, sm_120). Weights: `~/inkling-bench/models/` (sha256-verified at staging). Each power point has its own isolated config + results dir so no run overwrites another: **600 W** `config.titan.600w.json` → `results-600w/` (driver log `logs/suite-600w.log`, receipt `SUITE-DONE-600w.json`, finished 2026-11-22T18:09:03Z); **450 W** `config.titan.450w.json` → `results-450w/` (`logs/suite-450w.log`, `SUITE-DONE-450w.json`, finished 2026-11-22T21:02:18Z); both verify_exit 0. The accepted **350 W** baseline (report + 38 records + suite log) is preserved at `baseline-350w-accepted/`.

---

## Addendum — Strata native engine vs llama.cpp (v0.2.19, SQ-20, single card, 600 W)

**Status: verify gate PASS.** `python3 harness/verify.py results-strata/ --config harness/config.strata.json` → PASS (25 records: 24 measured, 1 blocked-informational — the 1M needle point, an engine OOM documented below). Output is isolated in `results-strata/`; the accepted llama.cpp `results-600w/` is untouched. Ran 2026-11-22/23 on the same Titan box at the same 600 W cap, median of 3 runs per point.

**What was tested.** The **GumbiiDigital/Strata** fork, tag **v0.2.19** (commit `c6f6427d9`), engine **0.1.65**, built from source on-box for **sm_120** (CUDA 12.8; CUDA 13.x is known to miscompile the IQ kernels on sm_120, issue #892). Pack: Inkling-2-Mini **SQ-20** (19,346,685,952 B, sha256-verified) + Vision shard (636,526,592 B), placed **fully in VRAM** via the repo's own `inkling_fit.place()` (experts 15.68 GB VRAM / 0 RAM / 0 SSD). Served by Strata's Python OpenAI-compatible server (`serve.server`). The harness gained a clean runtime abstraction (`cfg.runtime` selects `LlamaCppRuntime` or `StrataRuntime`; orchestration unchanged) so the same tracks drive either engine.

### Headline — faster where it runs, same accuracy, but cannot reach 1M

| Signal | llama.cpp 600 W | Strata v0.2.19 | Ratio |
|---|---:|---:|---|
| A decode, ctx-512 | 86.76 tok/s | 178.60 tok/s | **2.06×** |
| A decode, ctx-8k | 82.22 | 125.49 | 1.53× |
| A decode, ctx-32k | 77.69 | 92.86 | 1.20× |
| A decode, ctx-131k | 63.96 | 94.98 | 1.48× |
| E aggregate, 1 worker | 92.65 | 193.7 | **2.09×** |
| E aggregate, 4 workers | 228.28 | 298.7 | 1.31× |
| E aggregate, 16 workers | 519.81 | 583.0 | 1.12× |
| E aggregate, 64 workers | 906.77 | 1046.6 | 1.15× |
| H soak decode | 66.14 | 96.14 | 1.45× |
| H efficiency | 0.299 tok/W | 0.278 tok/W | 0.93× (llama better) |
| C combined eval | 0.80 | 0.80 | same |
| G vision | 0.625 | 0.625 | same |
| D needle, 1M point | runs (0/3 content) | **engine OOM — blocked** | Strata fails |

**Reading:** Strata's CUDA engine is markedly faster at single-stream and low-concurrency decode (2.06× at ctx-512, 2.09× at 1 worker), and the advantage narrows as context grows and as concurrency becomes batch/bandwidth-bound (1.15× at 64 workers). Accuracy is identical on evals and vision and comparable on needle retrieval through 262k. The clear weakness is **very long context**: the engine cannot prefill the 1M needle prompt on a single card (stall + CUDA OOM), a point llama.cpp completes. Per-point deltas: `results-strata/evidence/compare-strata-vs-llama-600w.txt`.

### Caveats — read before comparing the numbers

- **Thinking mode:** Strata v0.2.19's Inkling dialect **forces thinking off** (`enable_thinking=False`, effort "none"); there is no thinking-on arm. Strata is compared against the llama.cpp **thinking-off** arm only.
- **Reasoning is hidden:** Strata does not expose `reasoning_content` on this dialect (`reasoning_share=0`, `reasoning_tokens=0` on every record), whereas llama.cpp exposes it (`reasoning_share≈0.984`). Both engines still decode the full token budget; decode tok/s counts all predicted tokens either way, so the speed comparison is fair. But the needle `found_in_reasoning` diagnostic is **unmeasurable for Strata** (always 0) — an answer the model only "thinks" but does not speak scores 0 with no visibility.
- **Different quant, not a controlled same-weight test:** Strata runs **SQ-20** (19.3 GB); llama.cpp runs **Q4_K_XL** (17.6 GB). This is engine-vs-engine on each runtime's recommended pack, not an isolated quant comparison.
- **Prompt path differs:** Strata has no raw `/completion`; the harness maps completion to `/v1/chat/completions`, so the prompt is chat-template-wrapped. **Decode tok/s is comparable; prefill tok/s and TTFT are not directly comparable** (different prompt token counts). Strata prefill/TTFT are reported as measured, not ratioed against llama.cpp.
- **Single card:** Strata's published 131.6 tok/s headline is on **two** PRO 6000s. Every number here is **one** card. The single-card figure is the gap this job measures.
- **Power:** Strata's soak draws more power and runs hotter (below); part of its speed is higher draw, not only better kernels. Per-watt efficiency is comparable (llama.cpp marginally better).

### Track A — speed ladder (Strata, SQ-20, drafting off, thinking off)

| Context | Decode tok/s | Prefill tok/s | TTFT s | llama.cpp decode | Ratio |
|---:|---:|---:|---:|---:|---:|
| 512 | 178.60 | 3395 | 0.038 | 86.76 | 2.06× |
| 8,192 | 125.49 | 8477 | 0.921 | 82.22 | 1.53× |
| 32,768 | 92.86 | 6150 | 5.265 | 77.69 | 1.20× |
| 131,072 | 94.98 | 2878 | 45.413 | 63.96 | 1.48× |

Decode decays with context but less steeply than llama.cpp (178.6 → 125.5 → 92.9 → 95.0; the 32k/131k pair is within run-to-run noise). Prefill/TTFT are Strata-only measurements (chat-wrapped prompt; see caveats).

### Track C — evals (30 items, thinking off)

Math **0.70** (14/20), code **1.00** (10/10), combined **0.80** — identical to llama.cpp. 2685 solve tokens total; `reasoning_tokens=0` (hidden). Content parsing verified end-to-end (real graded answers, not empty).

### Track G — vision (8 items)

Accuracy **0.625** (5/8) — identical to llama.cpp. Strata's serve parses OpenAI `image_url` base64 parts; the Vision shard loads. Zero empty answers.

### Track E — concurrency (256-token answers, on-box load)

| Parallel | Strata agg tok/s | p50 s | p95 s | Failed | llama.cpp agg | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 193.7 | 1.32 | 1.32 | 0 | 92.65 | 2.09× |
| 4 | 298.7 | 3.42 | 3.42 | 0 | 228.28 | 1.31× |
| 16 | 583.0 | 6.93 | 7.01 | 0 | 519.81 | 1.12× |
| 64 | 1046.6 | 14.96 | 15.60 | 0 | 906.77 | 1.15× |

Zero failed workers at every level. Same shape as llama.cpp: low concurrency gains the most (clock-bound single-stream), high concurrency is batch/bandwidth-bound.

### Track H — soak / power (3 × 600 s continuous load)

| Window | Requests | Decode tok/s | Power mean W | Power max W | Temp max °C | tok/W |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 908 | 96.8 | 335 | 375 | 80 | 0.289 |
| 1 | 902 | 96.1 | 350 | 388 | 82 | 0.275 |
| 2 | 901 | 96.0 | 345 | 395 | 82 | 0.278 |

Median decode **96.14 tok/s**, **0.278 tok/W**, no fade (908/902/901 requests), no crashes. **Nuance:** Strata pushes the card harder than llama.cpp — mean power **345 W vs 221 W** (+56%), max temp **82 °C vs 66 °C**, and ~45% more requests/window (902 vs 621). Both stay below the 600 W cap (Strata max 395 W). The speed gain is partly higher power draw: per-watt efficiency is comparable, llama.cpp marginally better (0.299 vs 0.278 tok/W).

### Track J — DSpark speculative decoding

**Serve API (the harness path):** speculation shows **no measurable decode gain** and emits **no acceptance counters**:

| Context | Speculate off | Speculate on |
|---:|---:|---:|
| 512 | 155.47 tok/s | 155.53 tok/s |
| 32,768 | 90.45 | 90.46 |

Strata v0.2.19's serve solo DONE carries no `draft_n`/`draft_n_accepted` fields, so API acceptance is unmeasurable; the on/off arms are identical within noise.

**Native engine bench (receipt `~/inkling-bench/logs/strata-smoke.log`, Titan):** DSpark **does** work at the engine level and is **bit-exact**:

| Mode | tok/s | tok/iter | Parity vs plain |
|---|---:|---:|---|
| Plain decode | 128.7 | — | — |
| DSpark BLOCK-5 | 244.8 | 3.74 | 256/256 EXACT |
| DSpark BEAM-W4 TREE-6 (best) | **262.9** | 4.32 | 256/256 EXACT |
| DSpark BEAM-W4 ADAPT-3..12 | 262.5 | 3.76 | 256/256 EXACT |

Best DSpark **262.9 vs plain 128.7 = 2.04×**, every variant parity 256/256 EXACT. **Conclusion:** Strata's DSpark is a real, bit-exact ~2× engine-level gain, but the v0.2.19 **serve API path does not engage/report it** for these requests. This contrasts with llama.cpp, where J speculation was API-measurable but **negative** (50.21 vs 87.13 tok/s at ctx-512, acceptance 0.06). Acceptance/parity for Strata are cited from the native bench receipt, not the API.

### Track D — needle in a haystack (content hits out of 3)

| Context | depth | Strata | llama.cpp |
|---:|---:|---:|---:|
| 16,384 | 0.1 / 0.5 / 0.9 | 3/3 · 3/3 · 2/3 | 3/3 · 3/3 · 2/3 |
| 65,536 | 0.1 / 0.5 / 0.9 | 3/3 · 2/3 · 0/3 | 3/3 · 1/3 · 1/3 |
| 131,072 | 0.5 | 0/3 | 0/3 |
| 262,144 | 0.5 | 0/3 | 0/3 |
| 1,047,040 (1M) | 0.5 | **BLOCKED — engine OOM** | 0/3 (runs) |

Content-hit accuracy is **comparable** through 262k (Strata slightly better at 65k d0.5). Both engines stop emitting usable needle answers beyond ~65k. `found_in_reasoning` is 0 for Strata everywhere (reasoning hidden — see caveats); llama.cpp found the code in reasoning on 9/14 failed runs.

**The 1M engine failure (headline Strata weakness).** The 1M prompt is **1,010,991 tokens**. Strata prefilled to **267,776 tokens in 182 s, then froze with zero progress for ~7,092 s**, and the engine died with **CUDA out-of-memory** (`ink_cuda.cuh:54`, exit code 3). This is a hard deterministic stall, not slow prefill — a longer timeout cannot fix it. llama.cpp prefills the full prompt and completes the point. Evidence: `results-strata/evidence/D-1M-oom-serve.log` (868 lines, the stall + OOM) and `D-1M-oom-engine.log`. Recorded as a **blocked-informational** point with this reason; the verify PASS line prints it. **Strata cannot reach 1M context on a single PRO 6000 with the repo-recommended full-VRAM SQ-20 placement; llama.cpp can.**

### Tracks F / K

F (loop depth) **blocked** — loop count is baked into the pack; varying it needs re-exported packs (same as llama.cpp). K (two-node RPC) **excluded** — single card.

### Harness adapter (engineering note)

Clean runtime abstraction, backward compatible: `cfg.runtime` selects `LlamaCppRuntime` (default, byte-identical to the accepted baseline path) or `StrataRuntime`; `ServerService` orchestration unchanged. Wire mappings: completion → `/v1/chat/completions`; tokenize → `/v1/messages/count_tokens`; `n_predict` → `max_tokens`; track E parallel batch posts chat completions (Strata has no `/completion`); SIGTERM on the wrapper PID stops the engine; orphan sweep uses a bracketed `pgrep` pattern (no self-match). A `quant_override` config key stamps records **SQ-20** (tracks otherwise hardcode the llama.cpp `Q4_K_XL` label). **186 tests green** (165 existing + 21 new); strata and 600 W dry-runs exit 0. Worktree `.worktrees/strata-adapter` commit `d40d973` (base `42e6044` = pre-change backup); 12 files in `harness/`.

**Written by Cursor composer-2.5 + lead fix round. Checked by Sonnet quick check (FIX, 5 findings addressed).**

### Reproduce (Strata)

```bash
cd "Inkling 2 [RTX 6000]"
python3 -m pytest harness/tests -q                                   # 186 tests
# Titan at 600 W; SQ-20 pack + Vision staged sha256-verified; full_vram placement.
python3 -m harness.run --config harness/config.strata.json --lane 1  # tracks A C G E H J F D
python3 harness/verify.py results-strata/ --config harness/config.strata.json  # -> PASS
```

## Addendum — Upstream Strata PR arms (#6642 GGUF loader, #6520 NVFP4 official checkpoint)

**Status: verify gate PASS on both arms.** Ran 2026-11-23 on the same Titan box (RTX PRO 6000 Blackwell, sm_120), same 600 W cap, median of 3 runs per point, thinking off. Output isolated in `results-strata-gguf/`, `results-llama-gguf/`, `results-strata-nvfp4/`; the accepted `results-600w/`, `results-450w/`, `results-strata/` are untouched. Goal: test each upstream PR as a separate arm against the accepted llama.cpp 600 W baseline.

- `python3 harness/verify.py results-strata-gguf/ --config harness/config.strata.gguf.json` → PASS (10 records A/C/E/H)
- `python3 harness/verify.py results-llama-gguf/ --config harness/config.llama.gguf.json` → PASS (10 records A/C/E/H)
- `python3 harness/verify.py results-strata-nvfp4/ --config harness/config.strata.nvfp4.json` → PASS (10 records A/C/E/H)

**What was tested.** Two upstream Strata PRs, each on the engine build that actually carries it:

1. **#6642 (GGUF loader fix).** Proof: the v0.2.19 base engine *fails* the ggml-org mainline GGUF (`missing tensor loop_res_query.weight`); the cherry-pick (`dda0092b7`) fixes the loader. But GPU generation needs the newer GGUF CUDA backend, which only exists on current **main** — so the GGUF arm uses Strata main `v0.2.20-16-g6a6db7ff5` (contains the #6642 fix + GGUF CUDA backend), built sm_120. Weight: **ggml-org Inkling-2-Mini IQ4_NL** (17.6 GB), the exact mainline layout #6642 fixes. A **same-weight llama.cpp arm** (mainline build 12602, `418fe97b5` — the same build the accepted baseline used for IQ4_NL) runs the identical `.gguf` file for a controlled head-to-head.
2. **#6520 (NVFP4 official checkpoint).** Engine `strata-ink2` at PR head `v0.2.20-89-gbb137fd86`, built sm_120 on **CUDA 12.8**. Weight: the official **thinkingmachines/Inkling-2-Mini** NVFP4 checkpoint `@ed40e7da` (20.19 GB, 11 files SHA-256-verified). Resident placement (`--kv f16 --tc 1 --spec 1`, built-in MTP drafter + ViT).

### Headline — #6642 GGUF is parity-at-best; #6520 NVFP4 is fast + high-quality but does not batch

| Signal | llama.cpp 600 W baseline | #6642 Strata GGUF | #6520 Strata NVFP4 |
|---|---:|---:|---:|
| A decode, ctx-512 | 86.76 tok/s | 93.64 (IQ4_NL) | **163.06** (1.88×) |
| A decode, ctx-131k | 63.96 | 57.13 (IQ4_NL) | **96.02** (1.50×) |
| C combined eval | 0.80 | 0.533 (IQ4_NL) | **0.867** (math 0.80) |
| E aggregate, 64 workers | 906.77 | 126.6 (IQ4_NL) | 140.3 (**0.155×**) |
| H efficiency | 0.299 tok/W | 0.210 (IQ4_NL) | 0.262 |

**Reading:** The two PRs land very differently. **#6642** delivers GGUF *compatibility*, not speed: on the identical IQ4_NL file Strata's GGUF path is ~equal to llama.cpp at single-stream and worse at long context, quality, concurrency and efficiency (same-weight table below). **#6520** delivers a genuinely fast, higher-quality single-stream engine (1.5–1.9× the baseline, best eval score of any arm), but it **serializes concurrent requests** — aggregate throughput is flat ~140 tok/s at every concurrency level, so it is far slower than the baseline for throughput serving. Neither PR changes the accepted baseline.

### Arm 1 — #6642 GGUF: same-weight head-to-head (identical ggml-org IQ4_NL file)

The controlled comparison: both engines run the **same** `.gguf`. Strata main vs llama.cpp mainline b12602.

| Context | Strata GGUF decode | llama.cpp decode | Ratio S÷L |
|---:|---:|---:|---:|
| 512 | 93.64 | 95.37 | 0.98× |
| 8,192 | 88.34 | 88.04 | 1.00× |
| 32,768 | 79.88 | 81.32 | 0.98× |
| 131,072 | 57.13 | 63.58 | **0.90×** |

| Signal | Strata GGUF | llama.cpp GGUF |
|---|---:|---:|
| C combined (math / code) | 0.533 (0.30 / 1.00) | **0.733 (0.60 / 1.00)** |
| E agg, 1 / 4 / 16 / 64 | 88 / 158 / 96 / 127 | 92 / 244 / 552 / **978** |
| H soak decode | 62.6 tok/s | 70.0 tok/s |
| H power mean / temp | 299 W / 74 °C | 238 W / 69 °C |
| H efficiency | 0.210 tok/W | **0.293 tok/W** |

**Finding:** #6642's loader capability works (the file loads and generates on GPU), but Strata's GGUF path is **not a win** over llama.cpp on identical weights: ~equal single-stream, slower at 131k, lower math quality (0.30 vs 0.60), ~7× weaker at 64 workers, and worse per-watt. Strata's speed advantage lives in its **native SQ-20 pack** (prior addendum), not the GGUF compatibility path. Zero failed workers both engines. Per-point deltas: `evidence-pr-arms/compare-gguf-sameweight-strata-vs-llama.txt`.

### Arm 2 — #6520 NVFP4: official checkpoint vs accepted baseline

| Context | NVFP4 decode | Prefill tok/s | TTFT s | Baseline decode | Ratio |
|---:|---:|---:|---:|---:|---:|
| 512 | 163.06 | 2723 | 0.047 | 86.76 | **1.88×** |
| 8,192 | 143.02 | 8066 | 0.968 | 82.22 | 1.74× |
| 32,768 | 129.69 | 7212 | 4.490 | 77.69 | 1.67× |
| 131,072 | 96.02 | 4933 | 26.493 | 63.96 | 1.50× |

Quality: math **0.80** (16/20), code **1.00**, combined **0.867** — the **best of the three arms** (baseline and SQ-20 both 0.80, math 0.70). The official NVFP4 quant preserves more accuracy than SQ-20 / Q4_K_XL.

| Parallel | NVFP4 agg tok/s | p50 s | p95 s | Failed | Baseline agg | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 141.0 | 1.82 | 1.82 | 0 | 92.65 | 1.52× |
| 4 | 140.4 | — | — | 0 | 228.28 | 0.62× |
| 16 | 140.8 | — | — | 0 | 519.81 | 0.27× |
| 64 | 140.3 | 60.5 | 109.5 | 0 | 906.77 | **0.155×** |

**Concurrency is the clear weakness:** aggregate is **flat ~140 tok/s** at every level — `strata-ink2` has no batch slots and **serializes** requests (par-64 wall 116.8 s ≈ 64 × the 1.82 s single-stream latency). Fine for one user, poor for throughput serving.

H soak (3 × 600 s): median decode **91.50 tok/s**, **350 W** mean, **80 °C** mean, **0.262 tok/W**, no fade (846/858/865 requests), no crashes. Faster than the baseline (1.38×) but less power-efficient (0.262 vs 0.299 tok/W) and hotter — same pattern as SQ-20. Per-point deltas: `evidence-pr-arms/compare-nvfp4-strata-vs-baseline.txt`.

### Caveats — read before comparing

- **CUDA 12.8 vs the authors' CUDA 13.** `setup.py` rejects CUDA 12 for ink2 and `ink2_build` requests toolkit 13 for sm_120, citing a #220 prompt-path crash on 12.8. **Observed:** the engine builds (real sm_120 SASS), loads, and generates correctly on Titan's CUDA 12.8 through the full ctx-131072 ladder and the 600 s soak — no crash. The requirement is conservative for this card; treat it as a tested-on-12.8 result, not a vendor-supported configuration.
- **Different quants, not same-weight (NVFP4 arm).** NVFP4 official (20.19 GB) vs baseline Q4_K_XL (17.6 GB) vs SQ-20 (19.3 GB): engine-vs-engine on each path's recommended weights. The **GGUF arm IS same-weight** (identical IQ4_NL file both engines) and is the controlled comparison.
- **Prompt path differs** (chat-wrapped, no raw `/completion`): decode tok/s comparable; prefill/TTFT reported as measured, not ratioed (same caveat as the SQ-20 addendum).
- **Thinking off** throughout; ink2 reasoning is hidden (`reasoning_tokens=0`).
- **Single card**, 600 W cap. NVFP4 concurrency flatness is an engine design property (no batch slots), not a config artifact.

### Harness adapter (engineering note)

New runtime `strata-ink2` (`harness/strata_ink2_runtime.py`, `StrataInk2Runtime(StrataRuntime)`): overrides only runconfig generation (`arch: inkling2`, `--model DIR`, resident `--kv f16 --spec 1`) and the engine binary name; `serve.server` OpenAI/Anthropic layer, tokenize/complete/parallel_batch and orchestration inherit unchanged. The accepted `strata` and `llama.cpp` paths are byte-untouched. Registered in `server_runtime.py`.

### Reproduce (PR arms)

```bash
cd "Inkling 2 [RTX 6000]"
# Titan at 600 W. GGUF arm: Strata main + llama.cpp b12602 on the same ggml-org IQ4_NL file.
python3 -m harness.run --config harness/config.strata.gguf.json --lane 1   # A C E H
python3 -m harness.run --config harness/config.llama.gguf.json --lane 1    # A C E H (same weight)
# NVFP4 arm: strata-ink2 (CUDA 12.8 sm_120) + official checkpoint @ed40e7da (sha256-verified).
python3 -m harness.run --config harness/config.strata.nvfp4.json --lane 1  # A C E H
python3 harness/verify.py results-strata-gguf/  --config harness/config.strata.gguf.json   # -> PASS
python3 harness/verify.py results-llama-gguf/   --config harness/config.llama.gguf.json    # -> PASS
python3 harness/verify.py results-strata-nvfp4/ --config harness/config.strata.nvfp4.json  # -> PASS
```

Strata on Titan: source `~/inkling-bench/strata-src` (v0.2.19), build `~/inkling-bench/strata-build/strata-inkling` (0.1.65, sm_120), venv `~/inkling-bench/strata-venv`, pack `~/inkling-bench/models/strata-engine/Inkling-2-Mini-SQ-20.strata`. Driver log `logs/suite-strata.log`; receipt `SUITE-DONE-strata.json` (finished 2026-11-23T01:43:45Z, verify_exit 0 after the 1M point was documented as blocked). Base-first per the owner plan: PR cherry-picks #6642 (GGUF loader) / #6520 (NVFP4) are deferred until after this clean baseline. Nothing pushed to the fork without owner approval.