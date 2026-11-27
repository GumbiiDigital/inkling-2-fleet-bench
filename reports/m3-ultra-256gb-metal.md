# Inkling 2 Mini — Mac 3 Metal Benchmark Report

**Status: all measurement rows banked — Track D 1M (Thu 01:28Z) and Track B BF16 (Thu 01:57Z, PHASE2_COMPLETE) landed; `python3 harness/verify.py results/ --config harness/config-mac.json` PASS with zero blocked points. Claim-audit pending; flips to FINAL after the audit and owner Done card.**

Metal benchmark of Thinking Machines' **Inkling-2-Mini** on **Mac 3** (Mac Studio, M3 Ultra, 256 GB unified memory), replicating the DGX Spark CUDA suite with the same GGUFs, the same track shape, and a byte-identical shared harness. Cross-platform comparison against the Spark report is included where both sides' records are on disk.

## Setup

| Item | Value |
|---|---|
| Model | Inkling-2-Mini — 26.2B total / 3.4B active MoE, middle 12 of 24 layers looped 3×, 1M trained context, vision-capable |
| Host | mac3 (fabric 10.230.0.32) — M3 Ultra, 256 GB unified memory, Metal |
| Runtime A | Unsloth llama.cpp **b12588** (commit 883286076), macOS-arm64 CI artifact (run 289) — loads `inkling2` arch GGUFs (Q4_K_XL, Q8_K_XL, DSpark drafter, mmproj); Metal, draft-dspark present |
| Runtime B | Mainline llama.cpp **418fe97** built on-box (Metal) — loads `inkling2_loop` arch GGUFs (BF16, IQ4_NL) |
| Quants | unsloth/Inkling-2-Mini-GGUF @ ea0c79b (Q4_K_XL, Q8_K_XL); ggml-org/Inkling-2-Mini-GGUF @ db61c5b (BF16, IQ4_NL) — same files/revisions as the Spark job |
| Method | median of 3 runs per point; llama-server `/completion` streaming timings (`predicted_n/predicted_ms` authoritative for decode); per-point server lifecycle; evidence logs per record |
| Harness | `python3 -m harness.run --track <X> --config harness/config-mac.json`; finish gate `python3 harness/verify.py results/ --config harness/config-mac.json` |
| Drive host | Mac 1 over the fabric SSH mesh; long requests run detached on-box (M6, see engineering notes) |

**Architecture finding (same as Spark):** the two GGUF publishers use different architecture tags. Unsloth files (`inkling2`) load only on the Unsloth build; ggml-org BF16/IQ4_NL (`inkling2_loop`) load only on mainline. The harness selects the runtime per quant automatically. The Unsloth fork's macOS-arm64 prebuilt was recovered from CI and smoke-proven on Metal before any track ran (runtime provenance: `mac3:~/inkling-bench/runtime/MIX_PROVENANCE.txt`).

## Track A — Speed ladder (Q4_K_XL, mac3)

Decode tok/s, median of 3, thinking on/off via `chat_template_kwargs.reasoning_effort`:

| Context | Thinking off | Thinking on | Prefill tok/s | TTFT (s) |
|---:|---:|---:|---:|---:|
| 512 | 27.86 | 27.71 | ~320 | 0.44 |
| 8,192 | 26.41 | 26.43 | ~399 | 19.6 |
| 32,768 | 24.90 | 24.97 | 283 | 114.4 |
| 131,072 | 21.04 | 21.11 | 140 | 935.4 |

**Finding — the thinking toggle does not reduce reasoning (same as Spark).** `reasoning_share ≈ 0.984` at every point, on *and* off: ~98% of generated tokens go to `reasoning_content` even at effort "none". At the 128-token benchmark budget, content tokens = 0. Longer-budget evals (Track C) do produce answers.

## Track B — Precision ladder

Decode tok/s, median of 3 (5-item math accuracy is a small-sample signal, not a full eval):

| Quant | Runtime | @512 | @32k | Prefill @512 | Math @512 (5-item) |
|---|---|---:|---:|---:|---:|
| Q4_K_XL | unsloth b12588 (Metal) | 27.9 | 25.0 | 571 | 0.6 |
| Q8_K_XL | unsloth b12588 (Metal) | 26.5 | 24.0 | 573 | 0.6 |
| IQ4_NL | mainline 418fe97 (Metal) | 30.2 | 26.7 | 544 | 0.4 |
| BF16 | mainline 418fe97 (Metal) | 25.6 | 23.1 | 579 | 0.6 |

- **On Metal, IQ4_NL is the fastest rung at both contexts** (30.2 / 26.7) — a different ordering than Spark, where Q4_K_XL led (38.2 / 28.9 vs IQ4 36.9 / 28.0). Q8_K_XL is the slowest quantized rung here. BF16 is the slowest decode rung overall (25.6 / 23.1), but only ~8–9% behind Q4_K_XL — versus Spark's ~2× Q4-over-BF16 gap.
- Small-sample math (@512): Q4_K_XL, Q8_K_XL and BF16 0.6, IQ4_NL 0.4 — directionally similar to Spark (Q4/Q8 0.8, IQ4 0.4–0.6, BF16 0.6). IQ4_NL scored 0.8 at 32k: 5 items cannot rank quality firmly.
- **Prefill: no crossover in any matched-track data; the Spark report's crossover bullet is a cross-track artifact (flagged, correction routed to the Spark lead).** In spark-1's own B-track records, BF16 prefill is *slower* at both rungs: 519.8 vs 864.6 tok/s @512 and 221.4 vs 402.8 @32k (medians, `B-spark-1-*` records). The Spark REPORT.md B table's "Prefill @512" = 431 for Q4 is a **Track A** number (128-token prefill prompt; `A-spark-1` @512 = 431.4 / 432.1), not the B-track math prompt — different prompt shapes, not like-for-like; the crossover bullet (520 vs 431) rests on it. Candidate mechanism: A's short 128-token prefill vs B's longer math prompt (inference, not verified). Spark project files untouched by this job. On both Macs BF16 prefill is marginally *faster* at both rungs (mac3 579.1 vs 570.6 @512, 290.7 vs 285.1 @32k; m41 496.2 vs 481.9, 187.9 vs 187.0 — within ~3%). Runtime confound on every host: Q4 runs on unsloth-mix, BF16 on mainline; a runtime-matched Spark crossover test was never banked.
- Phase 2 executed 01:31–01:57Z (owner pick 1a): the 3 self-staged phase-1 quants deleted with byte-size guards; BF16 relayed from spark-1 via scp -3 (~90 s) with size + sha256 verification (7074c976… — independently cross-verified by the m41 lead against their relay manifest, the same fleet identity their B records ran against); track B rerun exit 0; superseded blocked records swept to `results/evidence/`.

## Track C — Evals (Q4_K_XL, mac3, thinking off)

| Suite | Score |
|---|---|
| Math (20 items) | **15/20 (0.75)** |
| Code (10 items) | **10/10 (1.0)** |
| Combined | 0.833 |

Budgets: math 1024, code 2048 tokens. **Identical combined score to Spark (0.833).** Per-run math 0.75 / 0.70 / 0.75, code 1.0 ×3; median reported.

## Track G — Vision (Q4_K_XL + mmproj, mac3)

| Suite | Score | Determinism | Misses |
|---|---|---|---|
| Vision (8 items) | **5/8 (0.625)** | identical across 3 runs, zero truncated answers | 3 shape miscounts (8 shapes → answers 7, 7, 6) |

**Same score and same miss pattern as Spark**, with byte-identical answers across runs (per-item audit: `results/evidence/G-items-mac3-2026-11-21T143455Z.log`). The misses are a genuine counting limitation of the model, not a harness artifact.

## Track H — Soak / thermal (Q4_K_XL, mac3, continuous load)

3 × 10-minute windows, back-to-back requests (229/window):

| Window | Decode tok/s | Thermal state (`pmset -g therm`) |
|---|---:|---|
| 0 | 24.41 | nominal |
| 1 | 24.39 | nominal |
| 2 | 24.40 | nominal |

**Flat throughput — no thermal throttling over 30 minutes.** No sudo on mac3, so GPU power/temperature numbers are not available (Spark's nvidia-smi wattage has no Mac equivalent here); the thermal-state readout and flat tok/s are the sustained-load proof. tok/s-per-watt is therefore not reported for Mac.

## Track J — DSpark speculative decoding (Q4_K_XL + drafter)

mac3, median of 3:

| Context | Baseline decode | Speculative decode | Ratio | Acceptance |
|---:|---:|---:|---:|---:|
| 512 | 27.68 | 16.00 | 0.58× | 7.1% |
| 32,768 | 24.94 | 13.94 | 0.56× | 10.5% |

**Fleet-wide negative result.** Every banked J record across all six hosts that ran the track is negative — all on the unsloth mix runtime (CUDA sm_121 and Metal), acceptance 6.7–13.5%, decode ratio 0.24–0.58× (table verified from each project's `results/` records, not summaries):

| Host | Hardware / backend | @512 ratio (acc) | @32k ratio (acc) |
|---|---|---|---|
| spark-1 | DGX Spark GB10 / CUDA | 0.46× (8.3%) | 0.54× (13.5%) |
| rtxspark-1 | RTX Spark 128 / CUDA | 0.45× (13.4%) | 0.47× (9.5%) |
| forge | RTX 5090 / CUDA | 0.54× (7.5%) | 0.58× (10.0%) |
| titan | RTX PRO 6000 / CUDA | 0.24× (6.7%) | 0.26× (9.6%) |
| mac3 | M3 Ultra / Metal | 0.58× (7.1%) | 0.56× (10.5%) |
| m41 | M4 Max / Metal | 0.57× (10.1%) | 0.48× (12.3%) |

The drafter's acceptance is far below break-even everywhere; draft+verify overhead dominates. DSpark does not help this model on any tested hardware/backend at these settings. No positive contrast record exists in any project. Record paths: `results/J-<host>-q4_k_xl--ctx-<ctx>-prefill_tokens-<pt>-speculate-{true,false}.json` in each project (forge's carry a `server_flags--ngl-999` suffix).

## Track F — Loop depth: BLOCKED (recorded, not failed)

The loop count is baked into the GGUF (`LLM_KV_LOOP_COUNT`); the shipped files cannot vary it at load time. Re-exported GGUFs with custom loop counts are required. Recorded as a blocked point with reason; verify treats it as informational.

## Track K — Two-node: N/A

Single-host project; the two-node RPC track is not applicable (recorded as blocked-informational with reason).

## Track D — Needle in a haystack (Q4_K_XL, mac3)

One fact per document, chat grading on the code token in content, temperature 0, 2048-token budget, needle never trimmed. Same documents/seeds as the Spark suite (config equality lead-verified).

| Context | Depth | Content accuracy (median) | Fact retrieved¹ | Clean stops |
|---:|---:|---:|---:|---:|
| 16,384 | 0.1 / 0.5 / 0.9 | 1.0 / 1.0 / 1.0 | 3/3 each | 3/3 each |
| 65,536 | 0.1 / 0.5 / 0.9 | 1.0 / 1.0 / 1.0 | 3/3 each | 3/3 each |
| 131,072 | 0.5 | 0.0 | 3/3 | 1/3 |
| 262,144 | 0.5 | 0.0 | 2/3 | 0/3 |
| 1,047,040 | 0.5 | 0.0 | 1/3 | 0/3 |

¹ "Fact retrieved" = the needle code appears in the answer **or** in the reasoning trace (secondary metric). "Clean stops" = `finish_reason=stop` within the 2048-token budget (raw finish reasons, not derived flags).

- **Perfect through 65k** (all depths, all runs clean, 281–452 tokens).
- **At 131k the answer budget is exhausted in 2/3 runs** (one clean 367-token answer; two `length` stops at ~1986 tokens with the fact found in reasoning). Spark was 3/3 clean at 131k — Mac's reasoning runs longer at this context. This is a measured divergence, not a harness difference (same documents, seeds, budget, grading).
- **At 262k all 3 runs hit `length`** (~1980 tokens); the fact is found in reasoning in 2/3 ([T,F,T]). The second Metal box (m41) shows the same 0/3 accuracy, 3/3 length, [T,F,T] pattern. Spark at 262k: fact retrieved 2/3, one clean stop.
- **1M: retrieval failure, cross-platform.** mac3 0/3 correct, all 3 runs `finish_reason=length` with **empty answers** — every generated token went to reasoning (8.0–8.7k chars); the fact surfaced in reasoning in only 1/3 runs ([F,F,T]). Runs 2–3 stopped at the context ceiling (1,984 tokens, see methods note); run 1 hit the 2048-token budget. Spark's 1M was also 0/3 all `length`, with degenerate repetition shown by an 8192-token probe — on neither platform is the failure a budget artifact. The full 1,048,576-token slot loads and runs: 3 runs × 12.5–12.8 h (prefill 22.86–23.14 tok/s average, decode 5.55–5.61 tok/s; totals 45,057.6 / 46,143.1 / 46,122.0 s).
- **Methods note — context-ceiling truncation (both Metal boxes, verified from stop lines + final eval lines):** runs 2 and 3 had identical prompts (1,046,592 tokens — both document trims boundary-active landed at the same count) and identical decode lengths (1,984): prompt + decode = exactly server_ctx 1,048,576 (`truncated = 1`). Run 1's word list exhausted ~12.6k under target (prompt 1,033,976), so the full 2048-token budget fit (`truncated = 0`). These are context-ceiling stops, not EOS. Rate measurements are unaffected.
- **Resolved — the "4-token prompt difference" was a log-line artifact:** the `prompt processing, progress = 1.00` line under-reports the final `prompt eval time` token count by exactly 4 on this build, in all three runs (progress finals 1,033,972 / 1,046,588 / 1,046,588 vs eval counts 1,033,976 / 1,046,592 / 1,046,592). The authoritative eval lines show run-2 prompts **identical across hosts** (1,046,592) and decode identical (1,984 — also the record's tokens field). An earlier draft of this note carried the difference as an open observation localized to the chat path; that was based on comparing a progress-line count (mac3) against an eval-line count (m41). All rate measurements used the eval lines. The within-run 4-token offset is a server logging detail (mechanism unverified, zero measurement impact). Confirmed symmetric by the m41 lead from their banked log (run 1: progress 1,033,972 vs eval 1,033,976; run 2: 1,046,588 vs 1,046,592) — observation closed in both reports; eval-line agreement on the rendered prompts is direct token-count evidence.

## Track E — Concurrency (Q4_K_XL, mac3, on-box load)

Aggregate output tok/s, median of 3, **zero failed workers at every level**:

| Parallel | Output tok/s | p50 latency | p95 latency | Server ctx |
|---:|---:|---:|---:|---:|
| 1 | 27.5 | 9.3 s | 9.3 s | 8,192 |
| 4 | 71.0 | 14.4 s | 14.4 s | 8,192 |
| 16 | 132.6 | 30.9 s | 30.9 s | 8,192 |
| 64 | 278.0 | 58.9 s | 58.9 s | 32,320 |

Throughput scales ~10× from 1→64 concurrent requests (27.5 → 278.0 tok/s); per-request latency rises as slots share the GPU. Load is generated on-box (local HTTP threads), the same design Spark adopted after its SSH fan-out attempt understated throughput.

## Cross-platform: Mac 3 Metal vs DGX Spark CUDA (Q4_K_XL)

Both sides' records on disk; Spark numbers from `Inkling 2 [SPARK]/REPORT.md` (FINAL, verify PASS).

| Metric | mac3 (M3 Ultra, Metal) | spark-1 (GB10, CUDA) | Note |
|---|---|---|---|
| Decode @512 | 27.9 | 38.5 | Spark 1.38× |
| Decode @8k | 26.4 | 32.9 | Spark 1.24× |
| Decode @32k | 24.9 | 28.7 | Spark 1.15× |
| Decode @131k | 21.0 | 19.6 | **Mac 1.07× — the gap closes and reverses** |
| Prefill @512 / @32k / @131k | ~320 / 283 / 140 | ~432 / ~397 / ~359 | Spark faster at every prefill point |
| BF16 decode @512 / @32k | 25.6 / 23.1 | 19.9 / 16.7 | Mac 1.29× / 1.39× |
| Concurrency @64 aggregate | 278.0 | 368.8 | Spark 1.33× |
| Evals (C) | 0.833 | 0.833 | identical |
| Vision (G) | 5/8, same misses | 5/8, same misses | identical, deterministic |
| Needle 131k | 1/3 clean | 3/3 clean | Mac reasoning exhausts budget earlier |
| Needle 262k | 0/3 acc, 2/3 found, 0/3 clean | 0/3 acc, 2/3 found, 1/3 clean | same failure mode |
| Needle 1M | 0/3, all length, empty answers | 0/3, degenerate at 8192 probe | retrieval failure cross-platform |
| Speculation (J) | 0.56–0.58× | 0.46–0.54× | negative fleet-wide (6 hosts) |

The second Metal box (m41, M4 Max 128 GB) runs the identical suite; its report is `Inkling 2 [M4 Max 128]/REPORT.md`. Verified cross-checks from its banked J baselines: decode 32.55 @512 / 26.13 @32k (faster than mac3 at small ctx; both Macs converge at long ctx). 1M run-1 server-log actuals (both leads, Wed): mac3 45,057.6 s vs m41 65,591.9 s = **1.456×** (attempt-3 pair: 1.458×; run-2 pair: 1.467× — the pace ratio is stable across attempts and transport changes). mac3 record LANDED Thu 01:28Z (3-run totals 45,057.6 / 46,143.1 / 46,122.0 s); m41 record expected ~Thu 19:45Z.

## Harness engineering notes

- Shared harness byte-identical with the m41 project (coordination via lead-to-lead channel before any change); mac-specific settings live in `config-mac.json`. Milestones M1–M6; **191 unit tests**; verify gate (≥3 runs, medians, evidence per approved track×host; blocked informational; any top-level failed record = FAIL).
- Runtime split mirrors Spark exactly: unsloth b12588-mix for Q4_K_XL/Q8_K_XL + J drafter; mainline 418fe97 for IQ4_NL/BF16.
- **1M attempts 1–3 failed, each root-caused and preserved** (`results/evidence/`): (1) stale floor-30 timeout cap from a config race — killed run 1 at 89%; (2) session teardown killed a nohup-only runner; (3) **D3: a network event killed the single silent SSH session holding the 12.5 h request at 80% prefill** (the same event killed m41's run 2 at 79%; Mac 1's own links did not flap — upstream). Runs 1–2 of D3 completed (45,050 s / 46,141 s) but per-point records write only after run 3, so the point repeated.
- **M6 detached transport (the fix):** requests with timeout > 600 s run detached on-box (`nohup sh -c` wrapper → stdout/stderr/exitcode files, atomic exitcode); the driver polls with short SSH calls and tolerates ~2 h of transient SSH failures; output is fetched with retries and deleted only after both files arrive. Two review rounds (4 bugs found and fixed: kill-pattern mismatch, launch AND-list hang, unchecked fetch+delete, nested quoting), 191 tests, live smoke, then **production proof: D4 held all three full 12.5–12.8 h requests through the overnight windows that killed D3, zero anomalies** — run 1: 45,057.6 s vs D3's 45,050 s (0.017%); run 2: 46,143.1 s vs D3's 46,141 s (0.005%); run 3: 46,122.0 s. Record landed Thu 01:28Z, exit 0, verify PASS; full 3-run server log preserved in `results/evidence/`.
- Durable runners: setsid double-fork (PPID 1), unbuffered logs, fail-closed config guards, self-contained completion status files; launchd not viable on the external volume (TCC). Session-bound watchers are an alert path only — correctness never depends on them.
- Lessons recorded in Forgejo memory (journal 2026-11-21…24): detach long compute from fragile transport; extrapolate ETAs from completed runs; recompute ETAs when quoted; records before claims.

## Reproduce

```bash
python3 -m harness.run --track A --config harness/config-mac.json
python3 -m harness.run --track D --config harness/config-mac-1m-retry.json   # 1M-only retry
python3 harness/verify.py results/ --config harness/config-mac.json          # finish gate
```

Records: `results/<track>-<host>-<params>.json`; evidence: `results/evidence/` (includes preserved failed attempts and full 1M server logs).

## Completion gate

1. ~~Track D 1M record~~ **LANDED Thu 01:28Z** (0/3, all length, empty answers; row + cross-platform framing filled above).
2. ~~Track B BF16~~ **LANDED Thu 01:57Z** (PHASE2_COMPLETE, B exit 0, verify PASS, zero blocked points; row + crossover check filled above).
3. Claim-audit → owner Done card → FINAL.