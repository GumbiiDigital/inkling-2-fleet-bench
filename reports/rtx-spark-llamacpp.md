# Inkling 2 Mini — RTX Spark single-box baseline report

Date: 2026-11-22 · Host: `eigenlabsrtxspark128gb-1` (alias `rtxspark-1`) · Method: identical to the SPARK job (`Inkling 2 [SPARK]`, REPORT.md), single box, tracks A–J (no K — it needs two nodes).

**Gate: `python3 harness/verify.py results/ --config harness/config.rtx.json` → PASS.** 37 records, zero track failures, suite ran 2026-11-21 21:40 UTC → 2026-11-22 06:48 UTC (~9 h 08 m), unattended.

## Hardware and software (measured, not assumed)

- NVIDIA RTX Spark N1X, Windows 11 ARM64, 6144-core Blackwell GPU, driver 616.62.
- **128 GB unified RAM** (SK Hynix LPDDR5x-9400; SMBIOS `Win32_PhysicalMemory` = 137,438,953,472 bytes), split by firmware: ~64 GiB visible to Windows, **~62.4 GiB (63,936 MiB) assigned to the GPU**. The GPU-side share is what constrains model size. Correction 2026-11-22: initial probes read only the OS/GPU views and misreported the box as 64 GB total; owner corrected this, re-verified on-box.
- `nvidia-smi` lists GPU 0 (RTX N1X) and GPU 1 (NPU/DLA); all telemetry queries GPU 0 only.
- Python 3.12.10 native ARM64 on-box; harness drives the box over SSH (Tailscale endpoint, ~4 ms), all timings measured on-box.
- Runtimes (prebuilt Windows ARM64 CUDA zips, sha256-verified):
  - unsloth llama.cpp **b12588-mix-c0c3de8** (commit 883286076) → `inkling2` arch (unsloth GGUFs).
  - mainline llama.cpp **b12608** (commit 93cb37ea7, contains SPARK's 418fe97) + CUDA 13.5 runtime → `inkling2_loop` arch (ggml-org GGUFs).
- Models (Hugging Face, pinned revisions, every file sha256-verified on-box; manifest `results/evidence/staging-models-manifest.json`):
  - `unsloth/Inkling-2-Mini-GGUF @ ea0c79b`: UD-Q4_K_XL (17.63 GB), UD-Q8_K_XL (22.87 GB), dspark drafter Q8_0 (0.40 GB), mmproj-F16 (0.64 GB).
  - `ggml-org/Inkling-2-Mini-GGUF @ db61c5b`: BF16 (52.23 GB), IQ4_NL (17.63 GB).
- Build/arch pairing proven live in smoke (`results/evidence/smoke-20261121T213456Z.json`, 8/8 checks): each build loads its own model family; server survives SSH session close (Windows job-object breakaway flag, proven on-box); port frees after stop.

## Method (same as SPARK)

Median of 3 runs per point; timings from llama-server on-box; evidence log per record; thinking toggle via `chat_template_kwargs.reasoning_effort` (`none`/`high`); temperature 0 where graded; blocked points recorded as blocked, never faked. Track F blocked-informational (loop count baked into GGUF). Full needle plan including 1,047,040 ctx.

## Track A — Speed ladder (Q4_K_XL)

Decode tok/s, median of 3:

| Context | Thinking off | Thinking on | Prefill tok/s | TTFT (s) |
|---:|---:|---:|---:|---:|
| 512 | 36.31 | 36.14 | ~245–268 | 0.58 |
| 8,192 | 31.68 | 31.53 | ~490 | 16.0 |
| 32,768 | 28.27 | 28.21 | ~372 | 87.1 |
| 131,072 | 20.37 | 20.19 | ~340 | 385.1 |

**The SPARK thinking-toggle finding reproduces exactly:** `reasoning_share ≈ 0.984` at every point, on *and* off; at the 128-token budget content tokens = 0. The toggle changes effort labeling, not behavior at this budget.

## Track B — Precision ladder

Decode tok/s, median of 3 (5-item math accuracy is a small-sample signal, not a full eval):

| Quant | Runtime | @512 | @32k | Prefill @512 | Math (5-item) |
|---|---|---:|---:|---:|---:|
| Q4_K_XL | unsloth b12588 | 37.37 | 29.24 | 601 | 0.4 |
| IQ4_NL | mainline b12608 | 38.37 | 28.64 | 538 | 0.6 |
| Q8_K_XL | unsloth b12588 | 30.03 | 24.59 | 591 | 0.4 |
| BF16 | mainline b12608 | 20.60 | 17.64 | 413 | 0.6 |

- Q4_K_XL / IQ4_NL are the speed sweet spots (~1.8× BF16 decode).
- **BF16 prefill at 32k is slower than quantized** (214 vs 378 tok/s — memory-bound), same direction as SPARK. **But the SPARK 512-ctx crossover does NOT reproduce here:** BF16 prefills slower than Q4 at 512 too (413 vs 601). Reported as measured.
- BF16 at 32k fits the ~62.4 GiB GPU-side share (~56.5 GB weights+KV) and ran clean.

## Track C — Evals (Q4_K_XL, thinking off)

| Suite | Score |
|---|---|
| Math (20 items) | **15/20 (0.75)** |
| Code (10 items) | **10/10 (1.0)** |
| Combined | 0.833 |

Identical scores to SPARK. Budgets: math 1024, code 2048 tokens. Code sandbox runs on the Mac driver (same as SPARK).

## Track D — Needle in a haystack (Q4_K_XL)

One fact per document, chat grading on the code token in content, temperature 0, 2048-token budget, needle never trimmed.

| Context | Depth | Content accuracy (median) | Fact retrieved¹ | Clean stops |
|---:|---:|---:|---:|---:|
| 16,384 | 0.1 / 0.5 / 0.9 | 1.0 / 1.0 / 1.0 | 3/3 each | 3/3 each |
| 65,536 | 0.1 | 1.0 | 3/3 | 3/3 |
| 65,536 | 0.5 | 0.0 | 3/3 | 1/3 |
| 65,536 | 0.9 | 0.0 | 3/3 | 0/3 |
| 131,072 | 0.5 | 0.0 | 2/3 | 1/3 |
| 262,144 | 0.5 | 0.0 | 3/3 | 1/3 |
| 1,047,040 | 0.5 | 0.0 | 1/3 | 0/3 |

¹ "Fact retrieved" = needle code in the answer **or** the reasoning trace. "Clean stops" = finished within the 2048-token budget.

**Headline: 1M context loads and runs within the RTX Spark's ~62.4 GiB GPU-side share — no OOM** (full 1,048,576 slot, Q4 weights + KV ≈ 59 GB). But retrieval degrades **earlier than on SPARK**: perfect only through 16k at all depths; from 65k mid/deep the reasoning grows and exhausts the answer budget (facts still found in reasoning); at 1M, 0/3 content hits, all `finish_reason=length`. SPARK was perfect through 128k and partial at 256k. This is a real behavioral difference between the two boxes/builds, reported as measured — not a harness artifact (same code, same seeds, same grading).

**1M retrieval probe (supplementary, definitive):** a single-run probe with an **8192-token** budget (4× standard; identical document/seed to standard run 0, temperature 0) also failed — `needle_accuracy 0.0`, `found_in_reasoning false`, full 8192 tokens consumed (`finish_reason=length`), 34,609 chars of reasoning, empty answer, 6085 s request. The trace shows **degenerate repetition**: only 49 unique words across 5,720 — the model loops over NATO filler words ("hotel, kilo, lima, juliet, india…") instead of searching for the fact. So the 1M failure is **not** an answer-budget artifact — coherent retrieval breaks down at 1M context (0/3 standard + 0/1 probe), the same failure mode SPARK showed. Evidence: `results/evidence/D-1M-probe-8192-rtxspark-1.json`.

## Track E — Concurrency (Q4_K_XL, on-box load)

Aggregate output tok/s, median of 3, **zero failed workers at every level**:

| Parallel | Output tok/s | p50 latency | p95 latency | Server ctx |
|---:|---:|---:|---:|---:|
| 1 | 37.0 | 6.9 s | 6.9 s | 8,192 |
| 4 | 92.8 | 11.0 s | 11.0 s | 8,192 |
| 16 | 210.9 | 19.1 s | 19.3 s | 8,192 |
| 64 | 374.5 | 43.5 s | 43.7 s | 32,320 |

Throughput scales ~10× from 1→64 (37.0 → 374.5 tok/s), matching SPARK's shape (36.8 → 368.8).

## Track F — Loop depth: BLOCKED (recorded, not failed)

Same as SPARK: loop count is baked into the GGUF (`LLM_KV_LOOP_COUNT`); shipped files cannot vary it at load time. Recorded with reason; verify treats it as informational.

## Track G — Vision (Q4_K_XL + mmproj)

| Suite | Score | Determinism | Misses |
|---|---|---|---|
| Vision (8 items) | **5/8 (0.625)** | identical across 3 runs, zero empty answers | 3 shape miscounts (8 shapes → answers 7, 7, 7); one item uses the full 1024-token budget |

Same score and same genuine counting limitation as SPARK (SPARK answers were 7, 7, 6). Per-item audit: `results/evidence/G-items-rtxspark-1-2026-11-22T060122Z.log`.

## Track H — Soak / power (Q4_K_XL, continuous load)

3 × 10-minute windows, back-to-back requests (~203/window):

| Window | Decode tok/s | GPU power mean | Temp mean | Temp max |
|---|---:|---:|---:|---:|
| 0 | 21.55 | 21.4 W | 53.4 °C | 59 °C |
| 1 | 21.53 | 23.1 W | 53.9 °C | 59 °C |
| 2 | 21.61 | 22.1 W | 53.5 °C | 55 °C |

**Flat throughput — no thermal throttling over 30 minutes**, cooler than SPARK (53–59 °C vs 62–64 °C). ~0.98 tok/s per watt as reported. **Power caveat:** `nvidia-smi power.draw` on this box reads low (idle showed 0.09 W); the 21–23 W figures are what the sensor reports, not verified wall power. Do not compare watts across the two boxes.

## Track J — DSpark speculative decoding (Q4_K_XL + drafter)

| Context | Baseline decode | Speculative decode | Acceptance |
|---:|---:|---:|---:|
| 512 | 37.78 | 17.06 | 13.4% |
| 32,768 | 29.33 | 13.77 | 9.5% |

**Negative result reproduces: speculation is ~2.2× slower.** Acceptance (9–13%) is far below break-even; draft+verify overhead dominates. Same conclusion as SPARK.

## Spark vs RTX comparison (labeled — NOT an equivalence claim)

Different hardware (DGX Spark GB10, 121 GB usable GPU memory vs RTX Spark N1X, 128 GB unified RAM with a ~62.4 GiB GPU-side share), different OS (Linux vs Windows ARM64), different mainline build (418fe97 vs b12608, 6 commits apart). Same model pins, same harness, same method. Numbers are medians of 3 from each job's own report.

| Metric | SPARK | RTX Spark | Note |
|---|---:|---:|---|
| A: Q4 decode @512 | 38.5 | 36.3 | RTX ~6% slower |
| A: Q4 decode @32k | 28.7 | 28.3 | near-parity |
| A: Q4 decode @131k | 19.6 | 20.4 | near-parity |
| A: prefill @32k | ~397 | ~372 | near-parity |
| B: BF16 decode @512 / @32k | 19.9 / 16.7 | 20.6 / 17.6 | RTX slightly faster |
| B: Q4 prefill @512 | 431 | 601 | RTX faster |
| B: BF16 prefill crossover @512 | yes (520 > 431) | no (413 < 601) | does not reproduce |
| C: evals combined | 0.833 | 0.833 | identical (math 0.75, code 1.0) |
| D: perfect retrieval through | 128k (7/7 points) | 16k all depths; 65k shallow only | RTX degrades earlier |
| D: 1M loads/runs | yes (~59 of 121 GB) | yes (~59 of ~62.4 GiB GPU share, no OOM) | both run; both fail retrieval (degenerate repetition, 8192-budget probe on each) |
| E: aggregate @par64 | 368.8 | 374.5 | near-parity |
| G: vision | 5/8 | 5/8 | same counting misses |
| H: soak decode | ~30.1 | ~21.6 | RTX slower under continuous load |
| H: temp mean/max | 62 / 64 °C | 53.5 / 59 °C | RTX cooler |
| J: speculation | ~2× slower | ~2.2× slower | same negative result |
| Thinking toggle | no effect (share 0.984) | no effect (share 0.984) | reproduces exactly |

**Summary:** raw single-stream decode and concurrency throughput are near-parity between the two boxes; evals and vision scores are identical. The RTX Spark's practical limits are the GPU-side memory share (~62.4 GiB of its 128 GB unified pool — BF16 and 1M ctx fit with only a few GB of headroom; the split is firmware-set) and long-context retrieval quality, which degrades from 65k where SPARK held to 128k. Soak throughput is lower on RTX (21.6 vs 30.1 tok/s) while running cooler. Watt figures are not comparable (sensor caveat above).

## Run integrity

- Harness ported with an OS-backend split; POSIX path byte-identical (146 legacy tests untouched and green; 183 total tests pass). Two worker briefs, each with an independent read-only review (brief 1: FIX, 3 findings fixed in one round; brief 2: SHIP first pass).
- Windows SSH session kills children by default — launcher uses `CREATE_BREAKAWAY_FROM_JOB`, proven live before the suite (smoke `ssh-survival` check).
- Staging defect found and fixed live: two download writers overlapped on Q4 after a restart; stray process stopped, partial deleted, Q4 re-downloaded clean and hash-verified. No other file affected (all sha256-checked).
- Suite stdout log is empty by design (runner prints only on skip/failure; there were none). Live progress was tracked in `results/evidence/remote-20261121T214025Z.log`.
- Evidence: every record carries its evidence path under `results/evidence/`; staging manifests, smoke reports, and the 1M probe live there too.
