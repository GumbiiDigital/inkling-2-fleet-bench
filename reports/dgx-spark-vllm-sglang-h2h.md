# Inkling 2 Mini on DGX Spark — vLLM vs SGLang head-to-head

> Status: FINAL — 2026-11-22. All tracks complete: the campaign covered 20 track/engine
> combinations; the 4 problem tracks (D and G on both engines) were repaired and re-passed.
> Median of 3 runs. Verify gate: PASS (`harness/verify.py results-h2h/ --config harness/config-h2h.json`).

## Bottom line

- **Both native NVFP4 engines beat the llama.cpp Q4 baseline** on speed at every context ≥ 8K, and prefill is **6–7× faster** at long context (128K prompt: ~50–58 s vs 364 s).
- **DSpark is the headline reversal**: on llama.cpp speculation made decode ~2× *slower*; on the native engines it is **3.2–3.8× faster** (vLLM 142 tok/s, SGLang 120 tok/s at short context, vs ~37–39 without).
- **SGLang wins concurrency** (842 vs 645 vs 369 tok/s total at 64 parallel) and long-context prefill; **vLLM wins single-stream DSpark decode** and efficiency (0.95 tok/W).
- Quality is close: math 15/20 (llama.cpp, vLLM) vs 13/20 (SGLang); code 10/10 everywhere.
- spark-2 cross-check reproduces spark-1 within 3.4% (largest gap: one concurrency point; all others ≤1.4%) — results are box-independent.
- **Full-1M needle retrieval is unproven on every engine tested**: llama.cpp 0/3, vLLM 0/3, SGLang 0/3 — on all three, reasoning ate the answer budget. At **262K both native engines score median accuracy 1.0** (2 of 3 runs hit each) vs baseline 1/3. The community's 1M PASS was at ~950K tokens, not full 1M (pilot-recorded figure).

## Setup

| Item | Value |
|---|---|
| Boxes | spark-1 (spark-ab31, measurement), spark-2 (spark-faf4, cross-check); GB10 sm_121, 121.7 GiB unified |
| Model | thinkingmachines/Inkling-2-Mini rev `1fb4134b9a74f9681054f4ef82e4bf1f458a2337`, native NVFP4 (19.53 GB + 0.64 GB mtp.safetensors), sha256 identical cross-box |
| Varloop | TML-d24-varloop-research rev `657b53ba5da7baefe9ff15dca5cd25f5bd3dbaa2` (spark-1 only) |
| vLLM image | `vllm/vllm-openai:inkling-2-mini` @ `sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e` (digest pinned at staging; "community-verified" per pilot notes) |
| SGLang image | `lmsysorg/sglang:inkling-2-mini` @ `sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de` (same digest both boxes) |
| Method | one engine at a time on spark-1 (fairest), identical prompts/params, median of 3 runs, load generated ON-BOX (sshd MaxSessions 4), no SSH TCP forwarding |
| Baseline | llama.cpp unsloth Q4_K_XL, same harness, same ladders (REPORT.md, 41 records) |

### Recipes (per engine, as configured and smoke-tested)

- **vLLM**: tp1, reasoning-parser inkling, auto tool choice + inkling parser. Tiers: standard util 0.80 (262K ctx); max-fp8 `--kv-cache-memory-bytes 79456894976 --kv-cache-dtype fp8` + util 0.85; max-bf16 85899345920 bytes + util 0.85. DSpark: `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}`.
  - **util 0.85 on max tiers is a gate-only workaround**: this build checks free memory against util (default 0.92 ≈ 111.95 GiB) at startup even when the KV byte budget is set; after long uptime spark-1 had ~9–12 GiB unreclaimable use (gate mechanism and figure diagnosed during repair; the gate-error log was not preserved). Probe proved equivalence: *"reserved 74.0 GiB … as specified by kv_cache_memory_bytes … does not respect the gpu_memory_utilization config"* — KV pool exactly 74.0 GiB (the community recipe's budget, per pilot notes).
- **SGLang**: `sglang serve`, tp1/ep1, fa4, bfloat16, page-size 128, xgrammar, `--enable-multimodal`, reasoning-parser inkling, tool-call-parser inkling2, MAX_JOBS=4. mem-fraction-static 0.45 (standard) / 0.75 (max). DSpark adds `--speculative-algorithm DSPARK --speculative-num-draft-tokens 8` + 4 SGLANG_OPT_* envs.
- Non-standard tiers drop page caches before start (guarded sudo); per-container memory watchdog kills at MemAvailable < 3 GiB.

## A — single-stream speed

Decode tok/s (median of 3):

| ctx | thinking | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| 512 | none | 38.53 | 37.85 | 37.91 |
| 512 | high | 37.46 | 37.81 | 37.85 |
| 8192 | none | 32.86 | 35.55 | 35.50 |
| 8192 | high | 32.84 | 35.53 | 35.53 |
| 32768 | none | 28.74 | 31.86 | 31.56 |
| 32768 | high | 28.45 | 31.96 | 31.48 |
| 131072 | none | 19.59 | 22.44 | 22.20 |
| 131072 | high | 19.49 | 22.41 | 22.26 |

TTFT seconds (prefill latency — the biggest gap):

| ctx | llama.cpp | vLLM | SGLang |
|---|---|---|---|
| 512 | 0.33 | 0.19 | 0.20 |
| 8192 | 14.3 | 2.5 | 2.3 |
| 32768 | 81.6 | 11.1 | 9.6 |
| 131072 | 363.6 | 58.3 | 50.7 |

TTFT cells merge thinking modes (mean of the none/high medians, 1 decimal). Native engines hold ~38 tok/s at short context and decay more slowly with context (+14% over baseline at 128K). SGLang prefill is ~9–14% faster than vLLM at 8K–128K; at 512 both are ≈0.2 s. Both are 6–7× faster than llama.cpp at 128K.

## C — quality (thinking off, identical requests)

| engine | math /20 | code /10 | reasoning tokens (median) | solve tokens (median) |
|---|---|---|---|---|
| llama.cpp | 15 | 10 | n/a (not reported) | 2696 |
| vLLM | 15 | 10 | 2181 | 2705 |
| SGLang | 13 | 10 | 2012 | 2441 |

Note: `reasoning_effort: none` still produces brief reasoning on both native engines (the Inkling template's "none" = 0.0 still thinks; behavior proven in the records, README attribution per pilot notes). Both engines received identical requests, so fairness holds; grading scores the answer content only.

## E — concurrency (total output tok/s)

| parallel | llama.cpp | vLLM | SGLang |
|---|---|---|---|
| 1 | 36.78 | 29.53 | 36.09 |
| 4 | 86.09 | 93.56 | 119.55 |
| 16 | 196.23 | 279.74 | 386.82 |
| 64 | 368.75 | 645.00 | 842.02 |

SGLang scales best: **2.28× the baseline at 64 parallel**; vLLM 1.75×. Observation: vLLM at parallel 1 (29.5) is below its own single-stream A decode (37.8) — reproducible on spark-2 (29.6), so engine-side, not box-side; the E batch path uses /v1/completions with the on-box batch script. SGLang's par-1 (36.1) is close to its A decode (37.9; different prompt shape).

## H — 30-minute soak

| engine | decode tok/s | power mean W | tok/W | temp mean °C |
|---|---|---|---|---|
| llama.cpp | 30.18 | 37.43 | 0.81 | 62.26 |
| vLLM | 29.27 | 30.79 | 0.95 | 64.05 |
| SGLang | 28.58 | 31.22 | 0.92 | 64.67 |

All three stable for 30 min (same 3×10-min continuous protocol as the baseline). Native engines draw ~6 W less and are ~15% more efficient per token; vLLM edges SGLang on tok/W.

## J — DSpark speculative decoding

Decode tok/s:

| ctx | spec | llama.cpp | vLLM | SGLang |
|---|---|---|---|---|
| 512 | off | 38.97 | 37.29 | 37.10 |
| 512 | on | 17.95 | **142.01** | **119.69** |
| 32768 | off | 29.26 | 31.32 | 30.98 |
| 32768 | on | 15.81 | **110.90** | **109.03** |

The pilot found DSpark ~2× slower on llama.cpp; on the native engines it is a **3.2–3.8× speedup**. vLLM acceptance rate 0.542 (ctx 512; only run 3 reported acceptance, runs 1–2 null). Run-to-run variance is high on 128-token answers (vLLM @512: 142 / 67.5 / 160) — median of 3 is the agreed statistic; acceptance depends on how predictable the answer text is.

## F — varloop arm (loop-depth / shared-KV research weights)

| engine | passes | share_cache | decode tok/s | math /20 |
|---|---|---|---|---|
| vLLM | 1 | no | 61.53 | 12 |
| vLLM | 2 | no | 44.24 | 10 |
| vLLM | 3 | no | 34.26 | 8 |
| vLLM | 3 | yes | 34.47 | 14 |
| SGLang | 1 | no | 60.97 | 11 |
| SGLang | 2 | no | 43.77 | 12 |
| SGLang | 3 | no | 33.98 | 12 |
| SGLang | 3 | yes | 34.03 | 12 |

Loop passes cost decode speed as expected (1→3 passes: ~61→34 tok/s, ≈1.8×). Shared-KV caching barely changes speed on either engine. Math scores at n=20 are noisy (8–14); no consistent shared-KV quality gain visible at this sample size (cookbook claims gsm8k 90.0 vs 88.5 on a larger eval — worker-reported figure, not re-verified here). DSpark was not used with non-default pass counts (refusal reported during the pilot), so F runs without the drafter.

## D — needle-in-haystack

Plan: ctx 262144 and 1047040, depth 0.5, 3 runs each, both engines, max-KV tiers. Statistic: median of 3 per-run accuracies; per-run hits shown for transparency. Answer budget: dynamic cap min(2048, server_ctx − prompt_tokens − 64) ≈ 1928–2048 tokens (SGLang returns 400 on overflow; llama.cpp truncates n_predict per the pilot lesson — the cap makes the request shape identical on both engines).

| engine | 262K (median acc / runs hit) | 1M = 1047040 (median acc / runs hit) |
|---|---|---|
| llama.cpp (baseline) | 1/3 runs hit | 0/3 (+ 8192-budget probe: 0/1, degenerate reasoning) |
| vLLM max-fp8 | **1.0** — 2/3 runs hit (hit / hit / budget-exhausted) | **0.0** — 0/3; all finish=length, empty answers; run 3 found the code in reasoning only |
| SGLang max | **1.0** — 2/3 runs hit (hit / budget-exhausted / hit) | **0.0** — 0/3; all finish=length, empty answers; runs 2–3 found the code in reasoning only |

- **Full-1M retrieval fails the same way on all three engines**: the model spends the entire ~2K answer budget on reasoning and never emits the answer (finish=length, empty content). On vLLM and SGLang the needle code sometimes appears inside the reasoning text — retrieval is not wholly absent, but the answer path starves. This is NOT the community result: their PASS was at **949,953 tokens (~950K)**, ~97K short of our full-1M document, with a different answer budget (community figure recorded in the pilot; source not re-verified in this job). Full 1M retrieval remains unproven on every engine tested here; no 950K re-probe was run (owner decision: report as measured).
- 262K is a clear native-engine win: median accuracy 1.0 on both (2 of 3 runs hit each) vs baseline 1/3. The missed run on each engine exhausted the answer budget on reasoning — the same failure mode, at lower frequency.
- First-attempt failures repaired: vLLM startup gate (util 0.85, probe-proven equivalent); SGLang 400 (dynamic n_predict cap). Repair run 2 (SGLang D only) finished 2026-11-22 14:54:57Z, attempt 1, exit 0, 10180 s; no watchdog or OOM events — SGLang mem-fraction 0.75 held through 262K and full-1M prefills.

## G — vision (8 synthetic items × 3 runs)

| engine | accuracy | empty answers | budget hits |
|---|---|---|---|
| llama.cpp | 5/8 (0.625) | 0 | 0 |
| vLLM | 4/8 (0.5) | 0 | 0 |
| SGLang | 4/8 (0.5) | 0 | 0 |

Identical scores across all 3 runs per engine; no starvation (the baseline job's old G defect). Small synthetic set — one item's difference, not a strong quality signal.

## Cross-check (spark-2, A + E)

| metric | vLLM spark-1 | vLLM spark-2 | SGLang spark-1 | SGLang spark-2 |
|---|---|---|---|---|
| A decode ctx512 high | 37.81 | 37.76 | 37.85 | 37.76 |
| A decode ctx131072 high | 22.41 | 22.40 | 22.26 | 22.24 |
| E total par1 | 29.53 | 29.60 | 36.09 | 36.13 |
| E total par64 | 645.00 | 666.63 | 842.02 | 853.01 |

Largest difference 3.4% (vLLM E par64, 645→667); all other points ≤1.4% — the head-to-head conclusions are not box-specific.

## Caveats & honesty notes

1. `thinking=none` still reasons briefly on both engines (template semantics); identical requests keep the comparison fair.
2. vLLM max-tier util 0.85 is a startup-gate workaround, probe-proven not to change the KV budget (log line quoted above).
3. SGLang mem-fraction 0.75 was unverified on Spark before this job; the D repair runs were its first full test — 262K and full-1M prefills completed on both engines without OOM or watchdog kill.
4. J decode variance is high (short answers, acceptance-dependent); medians of 3 reported, raw runs in the records.
5. F math n=20 is too small to resolve the cookbook's 1.5-point gsm8k gap (cookbook figure worker-reported, not re-verified here).
6. E par-1 vLLM dip is real and reproducible (both boxes), not explained yet.
7. Record metadata: early campaign records carried the harness default `runtime_version: unsloth-llama.cpp`; normalized post-hoc on 2026-11-22 to the engine image tag (77 records; migration script in scratch, idempotent; measurements, params and slugs untouched; verify PASS re-run after migration). True engine identity always in `params.engine`; image digests pinned in the staging markers.
8. Soak ran 30 min continuous (3×10-min windows), the same protocol as the baseline.
9. K (two-node) out of scope; TensorFold ruled out (no Inkling support in a local checkout grep; decision worker-recorded).
10. D 262K per-run hits are 2/3 on each native engine (one budget-exhausted miss each); the headline uses the agreed median-of-3 statistic (1.0), not a perfect-run claim.
11. Not checked yet (absence claims, kept not dropped): no 950K re-probe record exists (absence of a record cannot prove the action never happened); no public post (git has no remote — proven — but publication elsewhere is not auditable from this record).
12. Claim-audit round 1 (independent read-only auditor) flagged 5 errors, all corrected in this final: campaign step count ("23/27" → 20 combinations/23 attempts), "SGLang prefill consistently ~13%" (slower at 512), cross-check "≤3%" (max 3.4%), soak "stricter than baseline" (identical protocol), TTFT llama.cpp@32K 81.5→81.6. Community/cookbook/gate-mechanism figures are attributed as worker-reported where no primary artifact exists.

## Evidence

- Staging markers: `~/inkling-bench/logs/H2H-STAGING-DONE.json` on both boxes; scratch `h2h-staging-spark{1,2}.json` (weight sha256, image digests).
- Records: `.worktrees/harness/results-h2h/` (spark-1) and `results-h2h-cross/` (spark-2); per-run raw values + medians + evidence logs.
- Campaign state: scratch `h2h-campaign.json` (20 track/engine combinations, 23 attempts incl. retries: 16 clean passes, 1 blocked vllm-D, 3 failed D/G); repair run 1 `h2h-repair-run1.json`, run 2 `h2h-repair.json` (finished 2026-11-22T14:54:57Z, verify exit 0); per-track logs `h2h-campaign-*.log`, `h2h-repair-*.log`.
- Verify: `python3 harness/verify.py results-h2h/ --config harness/config-h2h.json` → **PASS** (2026-11-22, after repair run 2 and after metadata migration).
- Harness commits: B1 5d42a09, B2 75d5663, compat f1bbd02, repair 26f8f44, metadata e8d8875, results+report commit (this finalization).

## Reproduce

```bash
cd ".worktrees/harness"
PYTHONPATH=$PWD python3 -m harness.run --config harness/config-h2h.json --engine vllm --track A
PYTHONPATH=$PWD python3 -m harness.run --config harness/config-h2h.json --engine sglang --track A
python3 harness/verify.py results-h2h/ --config harness/config-h2h.json
```
