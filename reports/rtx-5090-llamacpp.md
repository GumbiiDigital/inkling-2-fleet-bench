# Inkling 2 Mini — Forge RTX 5090 CUDA Benchmark Report

FINAL (2026-11-22). Track B BF16 and the Track D 1M rung are resolved as BLOCKED with evidence (BF16: sm_120 runtime bug; D 1M: owner decision to stop at 256k after a measured ~10.4 h/run). Max clean all-GPU needle rung: **256k**. All other tracks complete; `python3 harness/verify.py results/` → PASS.

## Setup

- Host: Forge, NVIDIA GeForce RTX 5090 (32607 MiB VRAM), driver 595.99.02, CUDA 12.8, 93 GB system RAM.
- Model: Inkling-2-Mini (24 physical layers, middle 12 looped 3× → 48 logical; hybrid local/global attention).
- Runtimes (same pins as the Spark session, for comparability):
  - Unsloth `b12588-mix-c0c3de8` (commit `883286076`), **cuda12-portable** prebuilt bundle, sha256 `4c30d5e3…`. The cuda13-portable bundle silently fell back to CPU on this CUDA 12.8 host and was swapped out; GPU residency verified via nvidia-smi for every profile.
  - ggml-org mainline `418fe97b5` (build 12602), built on-box for CUDA 12.8 sm_120.
- Weights staged from spark-1/spark-2, sha256-verified per file (~111 GB).
- Harness: Mac control plane, SSH + rhttp on-box transport; 3 runs per point, median reported. Suite order A C G E H B D J F; ran 2026-11-21 15:25 UTC → 2026-11-22 00:35 UTC (9 h 10 m).
- Device profile (probe-verified at the exact context sizes used):
  - Base: `-ngl 999` (all 25 offload units on GPU).
  - BF16 (mainline): `-ngl 14 -ub 256` — partial-offload regime, owner-approved, labeled as such.
  - Needle ≥ 524288 ctx: `-nkvo -ctk q8_0 -ctv q8_0 -fa on -ub 48` (KV in system RAM, all layers GPU; 28668 MiB at ctx 1048576).
  - Max clean all-GPU rung with f16 KV: **262144** (28408 MiB, probe-verified).

## Track A — Speed ladder (Q4_K_XL, all-GPU)

| ctx | decode tok/s | prefill tok/s | TTFT s |
|---:|---:|---:|---:|
| 512 | 80.0 | 680 | 0.2 |
| 8192 | 75.7 | 1852 | 4.2 |
| 32768 | 70.2 | 1636 | 19.8 |
| 131072 | 57.2 | 1382 | 94.6 |
| 262144 * | 46.4 | 1164 | 224.8 |

\* 262144 is a Forge-only extension rung beyond the Spark ladder (all-GPU f16 KV fits; documented deviation).

Side-by-side with Spark (same weights, same pins): Forge decode 80.0 vs 37.5 tok/s at 512 (2.1×), 57.2 vs 19.5 at 128k (2.9×).

Thinking toggle: `reasoning_effort=none` does NOT disable thinking in this GGUF's chat template — both modes emit reasoning tokens (126 of 128 at ctx 512). Spark records show the identical behavior; shared model characteristic, not a Forge regression. On/off rows are therefore near-identical.

## Track B — Precision ladder

Math set: 5 questions × 3 runs per point; accuracy grades final content only.

| quant | ctx | decode tok/s | prefill tok/s | math acc (median) | solve tokens |
|---|---:|---:|---:|---:|---:|
| IQ4_NL | 512 | 89.3 | 1837 | 0.6 | 609 |
| Q4_K_XL | 512 | 81.8 | 1676 | 0.6 | 596 |
| Q8_K_XL | 512 | 74.7 | 1573 | 0.6 | 595 |
| IQ4_NL | 32768 | 75.5 | 1125 | 0.6 | 628 |
| Q4_K_XL | 32768 | 71.5 | 1650 | 0.6 | 522 |
| Q8_K_XL | 32768 | 66.7 | 1574 | 0.6 | 654 |
| BF16 | 512 | BLOCKED | — | — | — |
| BF16 | 32768 | BLOCKED | — | — | — |

Forge ordering differs from Spark: IQ4_NL decodes fastest here (89.3 vs Q4 81.8), where Spark had Q4 38.2 > IQ4 36.9 — kernel efficiency differs on sm_120. All three quantized rungs score identically on the math set (median 0.6).

**BF16: BLOCKED — runtime bug on sm_120, not a config problem.** Two distinct failures, both isolated with controlled probes (11–17, receipts in `results/evidence/bf16-blocked/`):

1. **Runtime OOM at 32k** despite load-time fit (31410 MiB): the mainline VMM compute pool grows during real prefill and OOM'd at 82%. Fixed by `-ub 256` (probe 11: real 30255-token prefill, 126.8 tok/s, peak 29338 MiB). Lesson: load-fit ≠ runtime-fit.
2. **Numerical corruption after long prefills**: on any BF16 partial-offload server that has processed a ≥10k-token prefill, ALL subsequent generation collapses to repeated tokens (chats emit 200× "?"; the prefill's own output is "GGGGGGGG") — including on brand-new request slots. The original suite's BF16@512 point shows the same signature (acc 0.0, every chat hit the 1024-token cap, decode 2.4 tok/s). Ruled out by probes: flash-attention (fresh 32k chats are clean with FA on AND off), CUDA-graph reuse (`GGML_CUDA_DISABLE_GRAPHS=1` verified active — 0 graph reuses — still corrupt), slot KV reuse (fresh slots affected), the CPU BF16 backend (all-CPU ngl0 coherent), and the offload split itself (IQ4_NL ngl14 clean). Remaining suspect: the VMM compute pool (`GGML_USE_VMM`, compile-time only; no env/CLI switch). The Unsloth mix fork cannot load the ggml-org BF16 GGUF at all. A no-VMM rebuild would deviate from the Spark runtime pin and was not attempted.

Spark BF16 is unaffected: unified memory keeps all layers GPU-side, so it never enters the partial-offload path (acc 0.4–0.8, 19.9 tok/s). The blocked records carry the full evidence chain; the quantized rungs above are the usable Forge precision data.

## Track C — Evals (Q4_K_XL, thinking off)

Score **0.833** (code 1.0, math 0.75), 30 items, 2672 solve tokens. Identical to Spark (0.833; code 1.0, math 0.75).

## Track G — Vision (Q4_K_XL + mmproj)

Accuracy **0.625** (8 items, 0 budget hits, 0 empty answers). Identical to Spark (0.625).

## Track E — Concurrency (Q4_K_XL, on-box load)

| workers | output tok/s | p50 latency s | p95 latency s | failed |
|---:|---:|---:|---:|---:|
| 1 | 85.5 | 3.0 | 3.0 | 0 |
| 4 | 197.4 | 5.2 | 5.2 | 0 |
| 16 | 403.4 | 10.1 | 10.1 | 0 |
| 64 | 601.7 | 27.1 | 27.2 | 0 |

Spark: 36.8 / 86.1 / 196.2 / 368.8 at the same worker counts. Forge scales 7.0× from 1→64 workers with zero failures; aggregate throughput 1.6–2.3× Spark.

## Track H — Soak / power (Q4_K_XL, continuous load)

3 × 600 s windows, 1433 requests total:

| window | requests | decode tok/s | temp max °C | power mean W |
|---:|---:|---:|---:|---:|
| 1 | 478 | 51.0 | 55 | 184.1 |
| 2 | 478 | 50.9 | 55 | 184.9 |
| 3 | 477 | 50.8 | 56 | 183.0 |

No drift, no failures; 0.277 tok/W. The 5090 holds sustained load at 55–56 °C and ~184 W (well under its power limit).

## Track J — DSpark speculative decoding (Q4_K_XL + drafter)

| ctx | speculate | decode tok/s | acceptance |
|---:|---|---:|---:|
| 512 | off | 82.7 | — |
| 512 | on | 44.4 | 7.5% |
| 32768 | off | 71.7 | — |
| 32768 | on | 41.5 | 10.0% |

Speculation is **1.7–1.9× slower** — same conclusion as Spark (~2× slower). Acceptance 7–10% is far below break-even for this drafter/model pair.

## Track D — Needle in a haystack (Q4_K_XL)

3 runs per point; needle = code retrieval, accuracy grades the answer.

| ctx | depth 0.1 | depth 0.5 | depth 0.9 |
|---:|---|---|---|
| 16384 | 3/3 | 3/3 | 3/3 |
| 65536 | 3/3 | 1/3 | 2/3 |
| 131072 | — | 3/3 | — |
| 262144 | — | 1/3 | — |
| 1048576 | — | BLOCKED¹ | — |

All misses at 64k/256k are **output-budget exhaustion** (~1980 tokens, answer never emitted), not retrieval failure — the needle is found in reasoning (`found_in_reasoning=true` on every run, including misses). 128k is clean 3/3.

¹ **1M rung: BLOCKED by owner decision (2026-11-22, stop at 256k).** A full 3-run `-nkvo` q8-KV attempt was launched and timed out (exit 1). The server log proves the 1M **prefill completed** — 1,033,972 tokens in 33,059 s = **9.18 h at 31.28 tok/s average** — but prefill speed degrades with depth (32.0 tok/s at 97% → 31.3 at 100%; the earlier 41.8 tok/s was only the 72%-depth figure). Decode then ran at **0.45 tok/s** (the `-nkvo` config keeps KV in CPU pageable RAM, read per token at 1M depth), reaching 968/2048 generated tokens when the 9.78 h chat timeout fired during decode. One full run ≈ 9.18 h prefill + up to 1.26 h decode (2048/0.45) ≈ **10.4 h**; ×3 ≈ **30 h**, versus the ~21 h approved. The model generated **coherently** at 1M (968 tokens, no garbage — unlike BF16), so the `-nkvo` q8-KV config is proven at full depth (probe 14 + this run); it is simply slower than estimated. The owner elected to stop at 256k rather than extend the budget ~9 h (raise the timeout) or switch to untested-at-depth GPU-KV. Evidence: `logs/suite-failures/D-1M-rerun1-server.log` (prefill progress 1.00 + decode 0.45 t/s + cancel), the blocked record, and `results/evidence/d-1m-blocked-narrative.md`.

The **256k f16-KV all-GPU rung (28408 MiB) remains the max clean all-GPU configuration**; the D ladder is clean through 256k (8 points, 3 runs each). Spark's 1M result was a genuine retrieval failure; Forge's 1M rung is blocked on duration cost, not retrieval.

## Track F — Loop depth: BLOCKED (recorded, not failed)

Loop count is baked into the GGUF (`LLM_KV_LOOP_COUNT`); varying it requires re-exported GGUFs (`custom_loop_ggufs`). Same blocked status as Spark.

## Track K — Two-node fabric: NOT APPLICABLE

Single box; no fabric peer. (Spark session ran K over 200G.)

## Harness engineering notes

- CPU-fallback trap: the cuda13-portable Unsloth bundle silently ran on CPU under CUDA 12.8; its "offloaded 25/25" log line is misleading. GPU residency must be verified with nvidia-smi VRAM (idle = 18 MiB).
- Load-fit ≠ runtime-fit: BF16 fit at load (31410 MiB) but the mainline VMM compute pool grew during real prefill and OOM'd at 82% progress. Fixes must be verified with a real prefill, not a health check.
- Correct-output ≠ fresh-server chat: BF16 partial offload answered correctly on fresh servers (FA on or off) but corrupted after any ≥10k prefill. Coherence gates must include a long prefill followed by a chat, not just a short prompt. The 1M `-nkvo` q8-KV config passed exactly this gate (probe 14: ctx 1048576, correct answer, 28668 MiB).
- `-fa` requires a value (on|off|auto) in this build; compute buffer at 1M ctx scales with `-ub` (default 512 → ~51 GB).
- Long-request transport: SSH ServerAliveInterval=30/CountMax=20; rhttp outer timeout = inner + 60 s.
- Thinking toggle ineffective in this GGUF template (both modes reason); identical on Spark.

## Reproduce

```
python3 -m harness.run --config harness/config.json          # full suite
python3 harness/verify.py results/                            # finish gate
```

Config: `harness/config.json` (commit e7c4168). Evidence: `results/evidence/` (incl. `bf16-blocked/` probe chain), probe receipts on Forge `~/inkling-bench/logs/`.