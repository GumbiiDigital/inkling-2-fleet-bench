# Inkling 2 Mini on RTX Spark — vLLM vs SGLang head-to-head (job 2)

> Status: FINAL — 2026-11-23. All tracks complete: 16 track/engine combinations
> (8 tracks × 2 engines), every one passed on the first attempt, zero retries,
> zero hard-stops. Median of 3 runs. Verify gate: **PASS**
> (`python3 harness/verify.py results-h2h/ --config harness/config-h2h.rtx.json`, 58 records).
> Campaign ran 2026-11-22 16:14 UTC → 2026-11-23 02:53 UTC (~10 h 39 m), unattended.

This mirrors the DGX Spark head-to-head (`Inkling 2 [SPARK]`, REPORT-H2H.md) on the
RTX Spark box, and compares against this box's own completed llama.cpp baseline
(job 1, REPORT.md). **It is a labeled comparison, not an equivalence claim** — the
three jobs differ in hardware revision, OS, and engine build (see Caveats).

## Bottom line

- **Both native NVFP4 engines beat the llama.cpp Q4 baseline** on single-stream decode
  at every context (+23–36%), on concurrency (up to +146% at 64 parallel), and on
  prefill at long context (TTFT 128K: ~53–63 s vs 385 s, ~6–7× faster).
- **DSpark is the headline reversal, reproduced exactly**: on llama.cpp speculation made
  decode ~2.2× *slower* (acceptance 9–13%); on the native engines it is **3.1–4.1× faster**
  (vLLM 183.5 tok/s, SGLang 140.3 tok/s at short context, vs ~44.6 without).
- **SGLang wins concurrency** (922.9 vs 728.6 vs 374.5 tok/s total at 64 parallel) and
  long-context prefill; **vLLM wins single-stream DSpark decode** at short context.
- **The native engines run the varloop arm (track F) that llama.cpp could not** (loop count
  baked into the GGUF). Loop passes cost decode speed (~1.8× from 1→3 passes); shared-KV
  caching barely changes speed; math scores match the SPARK varloop run exactly.
- Quality is close: code 10/10 everywhere; math 15/20 (vLLM, llama.cpp) vs 16/20 (SGLang).
  Vision 4/8 on both native engines (same counting limitation as SPARK; llama.cpp 5/8).
- **Full-1M needle retrieval fails the same way on every engine**: at 1,047,040 tokens both
  native engines score 0/3 — all `finish_reason=length`, empty answers, the answer budget
  spent on reasoning. At **262K both native engines score median accuracy 1.0** (2 of 3 runs
  hit each) vs the llama.cpp baseline's 0 content hits. This reproduces the SPARK result
  run-for-run, including which runs surface the code only inside the reasoning trace.
- **The 1M tier fits and runs on this box** (no OOM): vLLM max-fp8 holds a ~4.0M-token KV
  pool; SGLang at the owner-approved mem-fraction 0.95 holds 1.08M tokens. The full
  109.67 GiB unified pool is usable at full speed (bandwidth probe: 199 GB/s past the
  62 GiB Windows carve-out).

## Setup

| Item | Value |
|---|---|
| Box | `eigenlabsrtxspark128gb-1` (config host key `spark-1`, role lane1); NVIDIA RTX Spark N1X (MSI Prestige N16 Flip AI+ C1XMT), Windows 11 ARM64, 6144-core Blackwell (cc 12.1), driver 616.62 |
| Memory | 128 GB unified LPDDR5x-9400; CUDA/torch see a **109.67 GiB** usable pool. `nvidia-smi` in WSL shows only the ~62.4 GiB firmware carve-out view; allocations past it spill to WDDM shared memory at full speed (probe-proven) |
| Execution | Harness runs **inside WSL2 Ubuntu-24.04** (systemd, docker + nvidia-container-toolkit GPU passthrough). Local transport (`bash -lc`, `shutil.copyfile`); load generated on-box; no SSH TCP forwarding |
| Model | thinkingmachines/Inkling-2-Mini rev `1fb4134b9a74f9681054f4ef82e4bf1f458a2337`, native NVFP4 (19,526,019,168 B + 635,801,416 B mtp), sha256 identical to the SPARK pins |
| Varloop | TML-d24-varloop-research rev `657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2`, sha256 identical to SPARK |
| vLLM image | `vllm/vllm-openai:inkling-2-mini` @ `sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e` (digest pinned, same as SPARK) |
| SGLang image | `lmsysorg/sglang:inkling-2-mini` @ `sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de` (digest pinned, same as SPARK) |
| Method | one engine at a time (fairest), identical prompts/params, median of 3 runs, evidence log per record, blocked points never faked |
| Baselines | job 1 = llama.cpp unsloth Q4_K_XL on this same box (REPORT.md, 37 records); SPARK = native NVFP4 on DGX Spark (REPORT-H2H.md) |

### Recipes (per engine, as configured and smoke-tested)

- **vLLM**: tp1, positional model, `--reasoning-parser inkling --tool-call-parser inkling`.
  Tiers: standard util 0.80; **max-fp8 `--kv-cache-memory-bytes 79456894976 --kv-cache-dtype fp8`
  + util 0.85** (Spark-identical). DSpark: `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}`.
  - Fit probe proved `--kv-cache-memory-bytes` **overrides** `--gpu-memory-utilization`
    ("reserved 74.0 GiB … skipped memory profiling … does not respect the gpu_memory_utilization
    config"): KV pool exactly 74.0 GiB, **3,998,233 tokens**, max concurrency for a 1,048,576-token
    request = 3.81×. No config change needed for vLLM.
- **SGLang**: `sglang serve`, tp1/ep1, `--attention-backend fa4 --page-size 128 --grammar-backend
  xgrammar --enable-multimodal --tool-call-parser inkling2`, bfloat16. mem-fraction-static 0.45
  (standard) / **0.95 (max — owner-approved deviation from SPARK's 0.75, see Caveat 2)**. DSpark
  adds `--speculative-algorithm DSPARK --speculative-num-draft-tokens 8` + 4 SGLANG_OPT_* envs.
  - SGLang's hybrid sliding-window KV pool scales linearly with mem-fraction (~1.13M tokens per
    unit): 0.75→850,944 (<1M), 0.90→1,020,928 (<1M), **0.95→1,080,192 (≥1M, measured stable)**.
- Per-container memory watchdog kills at host MemAvailable < 3 GiB (never triggered).

## A — single-stream speed

Decode tok/s (median of 3, thinking `none`; `high` is within noise — the toggle is a no-op, see Caveat 5):

| ctx | RTX vLLM | RTX SGLang | RTX llama.cpp (job 1) | SPARK vLLM | SPARK SGLang |
|---|---|---|---|---|---|
| 512 | 46.0 | 44.7 | 36.3 | 37.85 | 37.91 |
| 8,192 | 43.2 | 41.3 | 31.7 | 35.55 | 35.50 |
| 32,768 | 38.0 | 37.7 | 28.3 | 31.86 | 31.56 |
| 131,072 | 26.3 | 27.2 | 20.4 | 22.44 | 22.20 |

TTFT seconds (prefill latency):

| ctx | RTX vLLM | RTX SGLang | RTX llama.cpp | SPARK vLLM | SPARK SGLang |
|---|---|---|---|---|---|
| 512 | 0.33 | 0.17 | 0.58 | 0.19 | 0.20 |
| 8,192 | 2.41 | 2.21 | 16.0 | 2.5 | 2.3 |
| 32,768 | 10.90 | 9.59 | 87.1 | 11.1 | 9.6 |
| 131,072 | 63.3 | 52.7 | 385.1 | 58.3 | 50.7 |

RTX native decode is **+23–36% over the llama.cpp baseline** and **+16–23% over SPARK native**
at every context. Prefill at 128K is ~6× faster than llama.cpp but **slightly slower than SPARK**
(vLLM 63.3 s vs 58.3; SGLang 52.7 vs 50.7) — RTX wins decode, SPARK wins long-context prefill
latency. SGLang prefill is faster than vLLM at 8K–128K on both boxes. Prefill @512 is noisy
(sub-second TTFT; vLLM `none` 394 vs `high` 760 tok/s across runs).

## C — quality (thinking off, identical requests)

| engine | math /20 | code /10 | reasoning tokens (median) | solve tokens (median) |
|---|---|---|---|---|
| RTX vLLM | 15 (0.75) | 10 | 2181 | 2705 |
| RTX SGLang | 16 (0.80) | 10 | 2164 | 2595 |
| RTX llama.cpp (job 1) | 15 | 10 | n/a | 2501 |
| SPARK vLLM | 15 | 10 | 2181 | 2705 |
| SPARK SGLang | 13 | 10 | 2012 | 2441 |

RTX vLLM is **identical to SPARK vLLM** (math 15, reasoning 2181, solve 2705 — same seed/prompts).
RTX SGLang math 16 vs SPARK 13 is run-to-run variance at n=20 (temperature 0 grading, but sampling
varies). Code 10/10 on every engine. `reasoning_effort: none` still produces reasoning on both
native engines (template semantics; identical requests keep the comparison fair).

## E — concurrency (total output tok/s, on-box load)

| parallel | RTX vLLM | RTX SGLang | RTX llama.cpp | SPARK vLLM | SPARK SGLang |
|---|---|---|---|---|---|
| 1 | 33.5 | 44.5 | 37.0 | 29.53 | 36.09 |
| 4 | 105.4 | 148.3 | 92.8 | 93.56 | 119.55 |
| 16 | 385.7 | 453.3 | 210.9 | 279.74 | 386.82 |
| 64 | 728.6 | 922.9 | 374.5 | 645.00 | 842.02 |

**Zero failed workers at every level on both engines.** SGLang scales best: **2.46× the llama.cpp
baseline at 64 parallel** (922.9 vs 374.5), vLLM 1.95×. RTX native is +10–13% over SPARK native at
par64. The vLLM par-1 dip (33.5 below its own A decode 46.0) reproduces the SPARK pattern
(29.5 below 37.8) — engine-side batch-path behavior, not box-side.

## H — 30-minute soak (3 × 10-min windows, continuous load)

| engine | RTX decode tok/s | requests | RTX temp/power | SPARK decode | SPARK tok/W | SPARK temp °C |
|---|---|---|---|---|---|---|
| RTX vLLM | 39.8 | 374 | unavailable (0.0) | 29.27 | 0.95 | 64.05 |
| RTX SGLang | 38.8 | 364 | unavailable (0.0) | 28.58 | 0.92 | 64.67 |
| RTX llama.cpp (job 1) | 21.6 | ~203/window | 53.5 °C / ~22 W | 30.18 | 0.81 | 62.26 |

Flat throughput over 30 minutes on both native engines — no throttling, no drift. RTX native soak
decode (39.8 / 38.8) is higher than SPARK native (29.3 / 28.6) and the llama.cpp baseline (21.6).
**Temperature and power are not measurable in WSL2** (`nvidia-smi` returns 0.0 — Caveat 4), so
tok/W and thermal comparisons against SPARK are not possible for this job. Same 3×10-min protocol
as both baselines.

## J — DSpark speculative decoding (the headline reversal)

Decode tok/s:

| ctx | spec | RTX vLLM | RTX SGLang | RTX llama.cpp | SPARK vLLM | SPARK SGLang |
|---|---|---|---|---|---|---|
| 512 | off | 44.6 | 44.7 | 37.78 | 37.29 | 37.10 |
| 512 | on | **183.5** | **140.3** | 17.06 | **142.01** | **119.69** |
| 32,768 | off | 37.1 | 36.9 | 29.33 | 31.32 | 30.98 |
| 32,768 | on | **134.6** | **136.1** | 13.77 | **110.90** | **109.03** |

Speedup with speculation: RTX vLLM **4.1×** @512 / 3.6× @32K; RTX SGLang **3.1×** @512 / 3.7× @32K.
The llama.cpp baseline was **2.2× slower** (acceptance 9–13%, far below break-even). The native
engines reverse it: acceptance SGLang 0.344 @512 / 0.703 @32K, vLLM 0.885 @32K (vLLM @512 acceptance
not captured in the median run; smoke measured 0.54, matching SPARK's 0.542). RTX DSpark decode is
+17–29% over SPARK. Run-to-run variance is high on 128-token answers (acceptance depends on how
predictable the answer text is); median of 3 is the agreed statistic, raw runs in the records.

## F — varloop arm (loop-depth / shared-KV research weights)

**llama.cpp could not run this track** (loop count baked into the GGUF, recorded blocked in job 1).
The native engines run it. Decode tok/s at ctx 32,768:

| engine | pass 1 | pass 2 | pass 3 | pass 3 shared-KV | math /20 (1/2/3/3-shared) |
|---|---|---|---|---|---|
| RTX vLLM | 68.3 | 47.0 | 37.1 | 36.9 | 12 / 10 / 8 / 14 |
| RTX SGLang | 69.4 | 48.7 | 38.2 | 38.1 | 11 / 11 / 11 / 12 |
| SPARK vLLM | 61.53 | 44.24 | 34.26 | 34.47 | 12 / 10 / 8 / 14 |
| SPARK SGLang | 60.97 | 43.77 | 33.98 | 34.03 | 11 / 12 / 12 / 12 |

Loop passes cost decode speed as expected (1→3 passes ≈ 1.8×). Shared-KV caching barely changes
speed on either engine. **RTX vLLM math scores match SPARK vLLM exactly** (12/10/8/14 — same
seed/prompts). Math at n=20 is noisy (8–14); no consistent shared-KV quality gain visible at this
sample size. At ctx 512: RTX vLLM 81.8 / 58.5 / 46.4 tok/s (1/2/3 passes), SGLang 80.7 / 59.2 / 45.5.
DSpark was not used with non-default pass counts (F runs without the drafter, as on SPARK).

## D — needle-in-haystack

Plan: ctx 262,144 and 1,047,040, depth 0.5, 3 runs each, both engines, max-KV tiers. Statistic:
median of 3 per-run accuracies; per-run hits shown for transparency. Answer budget: dynamic cap
min(2048, server_ctx − prompt − 64) ≈ 1928–2048 tokens (identical request shape on both engines).

| engine | 262K (median acc / runs hit) | 1M = 1,047,040 (median acc / runs hit) |
|---|---|---|
| RTX vLLM max-fp8 | **1.0** — 2/3 hit (hit / hit / budget-exhausted) | **0.0** — 0/3; all finish=length, empty answers; run 3 found code in reasoning only |
| RTX SGLang max (0.95) | **1.0** — 2/3 hit (hit / budget-exhausted / hit) | **0.0** — 0/3; all finish=length, empty answers; runs 2–3 found code in reasoning only |
| RTX llama.cpp (job 1) | 0.0 content (fact in reasoning 3/3) | 0.0 (degenerate NATO-filler repetition; 8192-budget probe also 0) |
| SPARK vLLM max-fp8 | 1.0 — 2/3 hit | 0.0 — 0/3; run 3 found code in reasoning only |
| SPARK SGLang max (0.75) | 1.0 — 2/3 hit | 0.0 — 0/3; runs 2–3 found code in reasoning only |

- **Full-1M retrieval fails identically on all three engines**: the model spends the entire ~2K
  answer budget on reasoning and never emits the answer (`finish_reason=length`, empty content,
  `budget_hit=true`). On the native engines the needle code sometimes appears inside the reasoning
  text — retrieval is not wholly absent, but the answer path starves. **This reproduces the SPARK
  result run-for-run** (same 2/3 at 262K, same 0/3 at 1M, same runs surfacing the code in reasoning).
  It is a model behavior at 1M context, not an engine or memory limit. The community's reported 1M
  PASS was at ~950K tokens, not full 1M (SPARK pilot figure; not re-verified here).
- **262K is a clear native-engine win**: median accuracy 1.0 on both (2/3 runs hit each) vs the
  llama.cpp baseline's 0 content hits. The missed run on each engine exhausted the answer budget on
  reasoning — the same failure mode, at lower frequency.
- **The 1M tier fits and runs on RTX** (no OOM, no watchdog kill): vLLM max-fp8 (~4.0M-token pool)
  and SGLang max at 0.95 (1.08M-token pool) both completed the full-1M prefill. The llama.cpp
  baseline degraded earlier (perfect only through 16K; 262K content 0) — the native engines hold
  perfect median retrieval to 262K, matching SPARK.

## G — vision (8 synthetic items × 3 runs)

| engine | accuracy | empty answers | budget hits |
|---|---|---|---|
| RTX vLLM | 4/8 (0.5) | 0 | 0 |
| RTX SGLang | 4/8 (0.5) | 0 | 0 |
| RTX llama.cpp (job 1) | 5/8 (0.625) | 0 | 0 |
| SPARK vLLM / SGLang | 4/8 (0.5) | 0 | 0 |

Native engines score 4/8 on both boxes (identical across 3 runs per engine; no starvation). The
llama.cpp baseline scored 5/8. Small synthetic set — one item's difference, the same genuine shape
counting limitation documented in job 1 and SPARK, not a strong quality signal.

## RTX native vs SPARK native vs RTX llama.cpp (labeled — NOT equivalence)

Different hardware revision (RTX Spark N1X consumer MSI laptop, LPDDR5x-9400, 109.67 GiB usable
CUDA pool vs DGX Spark GB10, 121.7 GiB), different OS (Windows 11 ARM64 + WSL2 vs native DGX OS
Linux), different driver (616.62). Same model pins, same image digests, same harness method, same
seeds (host key `spark-1` → seed 11001, exact SPARK prompt parity).

| Metric | RTX native (job 2) | SPARK native | RTX llama.cpp (job 1) | Note |
|---|---|---|---|---|
| A decode @512 (vLLM) | 46.0 | 37.85 | 36.3 | RTX native fastest |
| A decode @131K (vLLM) | 26.3 | 22.44 | 20.4 | RTX native +17% vs SPARK |
| A TTFT @131K (vLLM) | 63.3 s | 58.3 s | 385.1 s | SPARK prefill slightly faster |
| C evals (vLLM) | 0.833 | 0.833 | 0.833 | identical |
| D 262K median acc | 1.0 (2/3) | 1.0 (2/3) | 0.0 content | native win, mirrors SPARK |
| D 1M | 0/3, runs | 0/3, runs | 0/3, degenerate | all fail retrieval; all load+run |
| E par64 (SGLang) | 922.9 | 842.02 | 374.5 | RTX native +10% vs SPARK |
| F varloop | runs | runs | blocked | native-only capability |
| G vision | 4/8 | 4/8 | 5/8 | native identical across boxes |
| H soak decode (vLLM) | 39.8 | 29.27 | 21.6 | RTX native highest |
| H temp / tok-W | unavailable | 64 °C / 0.95 | 53.5 °C / ~0.98 | WSL2 telemetry gap |
| J DSpark @512 (vLLM) | 183.5 (4.1×) | 142.01 (3.8×) | 17.06 (2.2× slower) | reversal on native, both boxes |
| Thinking toggle | no-op (0.984 / 1.0) | no-op (0.984) | no-op (0.984) | reproduces exactly |

**Summary:** the RTX Spark mirrors the DGX Spark head-to-head closely — same quality, same 262K
success, same 1M retrieval failure (run-for-run), same DSpark reversal, same varloop behavior, same
vision counting limit. The RTX native engines are generally **faster on decode and concurrency**
(+10–23%) but **slightly slower on 128K prefill latency**; these are different boxes/OS/drivers, so
no equivalence is claimed. The two practical RTX-specific facts: the usable CUDA pool is 109.67 GiB
(of 128 GB unified, full-speed past the firmware carve-out), and WSL2 cannot report GPU temperature
or power.

## Caveats & honesty notes

1. **OS difference**: job 2 runs inside WSL2 on Windows ARM64; SPARK runs native Linux; job 1 runs
   native Windows. The WSL2 layer is a real difference (virtualization, WDDM shared memory). No
   equivalence claim across the three.
2. **SGLang max-tier mem-fraction 0.75 → 0.95 (owner-approved 2026-11-22)**: SPARK used 0.75 and its
   ~121.7 GiB pool held 1M; the RTX 109.67 GiB pool with SGLang's hybrid sliding-window sizing held
   only 850,944 tokens at 0.75 (<1M, request would be rejected). Measured linear scaling: 0.90 →
   1,020,928 (still <1M), 0.95 → 1,080,192 (≥1M, stable). The owner chose to raise it so both engines
   run the full 1M comparison. **Only SGLang's memory-budget fraction differs from SPARK**; model,
   parsers, DSpark config and fa4 backend are identical. vLLM max-fp8 is Spark-identical.
3. **vLLM max-tier util 0.85** is Spark-identical; the fit probe proved `--kv-cache-memory-bytes`
   overrides util (KV pool exactly 74.0 GiB regardless), so it does not change the KV budget.
4. **Temperature and power unavailable in WSL2**: `nvidia-smi` returns 0.0 for temp/power inside
   WSL2. Track H reports decode stability only; tok/W and thermal comparisons vs SPARK/job 1 are not
   possible for this job. (Job 1 native Windows reported 53–59 °C; SPARK 62–64 °C.)
5. **Thinking toggle is a no-op**: `reasoning_effort: none` still reasons on both native engines
   (reasoning_share vLLM 0.984, SGLang 1.0). Identical requests keep the comparison fair; grading
   scores answer content only.
6. **J decode variance is high** (short 128-token answers, acceptance-dependent); medians of 3
   reported, raw runs in the records. vLLM @512 acceptance was not captured in the median run.
7. **F math n=20 is too small** to resolve a shared-KV quality gap; scores are noisy (8–14).
8. **E par-1 vLLM dip is real** (33.5 below A decode 46.0), reproduces the SPARK pattern; engine-side
   batch path, not explained further here.
9. **1M retrieval unproven on every engine** (model behavior, not memory): the community PASS was at
   ~950K, not full 1M (SPARK pilot figure, not re-verified). No 950K re-probe was run here.
10. **RTX decode faster than SPARK** is a measured difference between boxes (HW revision / driver /
    LPDDR5x-9400 memory), not a like-for-like equivalence; prefill @128K is slightly slower on RTX.

## Evidence

- Records: `h2h/results-h2h/` on Mac (58 JSONs, per-run raw values + medians + evidence paths);
  full evidence (5544 files, 1.3 GB) preserved on-box at `/root/inkling-h2h/h2h/results-h2h/evidence/`,
  integrity manifest `h2h/results-h2h/evidence-sha256.txt` (sha256 of every evidence file).
- Smoke evidence (pre-campaign validation): `h2h/smoke-evidence/smoke-evidence.tar.gz`
  (sha256 `4aebd50ae8d05c82cad2f84486f615b52295406c559d627b07b4632cd4670181`, matches box).
- Fit probes: `h2h/smoke-evidence/extracted/fit-{vllm-max-fp8,sglang-max,sglang-max-frac0.90,sglang-max-frac0.95}.json`.
- Staging: box `/root/inkling-h2h/weights/staging-manifest.json` (weight sha256 vs SPARK pins),
  `stage_markers.txt` (ALL:DONE), `bw_probe_result.json` (no carve-out speed cliff).
- Campaign state: scratch `rtx-h2h-campaign.json` (16/16 done, verify PASS, finished
  2026-11-23T02:52:59Z), log `rtx-h2h-campaign.log`, verify `rtx-h2h-verify.txt`.
- Verify: `python3 harness/verify.py results-h2h/ --config harness/config-h2h.rtx.json` → **PASS**.
- Harness commits: adaptation merge d7ef883 (harness 53ac800); config 0.95 + driver fix 36f53dd;
  smoke archive + launch 2a5ce81; results + report (this finalization).

## Reproduce

```bash
# inside WSL2 Ubuntu-24.04 on the RTX box
cd /root/inkling-h2h/h2h
python3 -m harness.run --config harness/config-h2h.rtx.json --engine vllm --track A
python3 -m harness.run --config harness/config-h2h.rtx.json --engine sglang --track A
python3 harness/verify.py results-h2h/ --config harness/config-h2h.rtx.json
```