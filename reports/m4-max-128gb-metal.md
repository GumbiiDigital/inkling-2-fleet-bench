# Inkling-2-Mini benchmark report — M4 Max 1 (`m41`, 128 GB)

**Status: DRAFT (campaign in flight).** Banked tracks final (A B C E G H, D ≤ 262k,
J, F/K); only the D-1M point and the final verify gate pending (attempt-4 retry in
flight). Companion runs: `Inkling 2 [M3U 256GB]` (mac3) and the Spark fleet reference.
Harness byte-identical to the 256 project (191 tests green since the M6 detached-transport
fix, 2026-11-24); only `harness/config-m41.json` / `config-m41-1m-retry.json` differ
(host paths, `prefill_tok_s_floor: 10`, retry needle_plan).

- Host: M4-MAX-1.local, Mac16,5, M4 Max 16c (12P+4E), 128 GB unified, macOS 27.2.1.
- Campaign: launched 2026-11-21T18:01Z under owner "run overnight unattended" authority.
  Attempt 1 → controlled stop 19:24Z (adopt shared timeout fix) → attempt 2 → controlled
  stop 23:15Z (widen patience floor) → attempt 3 (resume2) from 23:18Z. Zero hardware
  faults; all stops were config-adoption boundaries at record edges.
- Seed 33003; 3 runs per point, medians reported; PREFILL_HEADROOM 384.

## Runtimes and build (parity split, live-proven)

| Leg | Binary | sha256 (prefix) | Used for |
|---|---|---|---|
| unsloth b12588-mix | `runtime/mix-b12588-macos-arm64/llama-server` | `9c99f4b9…` | Q4_K_XL, Q8_K_XL + tracks A C D E G H J |
| mainline 418fe97 | `runtime/bin-path.txt` target | `e821e86c…` | ggml-org IQ4_NL, BF16 (track B) |

Mix binary copied mac3→m41 sha256-identical. No-sudo build: CLT-only
(`DEVELOPER_DIR=/Library/Developer/CommandLineTools`), `-DGGML_METAL_EMBED_LIBRARY=ON
-DLLAMA_CURL=OFF`, uv Python 3.12.13 venv. Metal in use proven per leg by verbose server
log (`ggml_metal_device_init`, embedded library compile) + `--list-devices` (MTL0 Apple
M4 Max). Smoke v3: 16/16 PASS both legs (`results/smoke-m41.json`).

## Track A — speed ladder (Q4_K_XL, mix runtime; medians of 3)

| ctx | mode | prefill tok/s | decode tok/s | TTFT s |
|---|---|---|---|---|
| 512 | no-think | 273.7 | 33.0 | 0.51 |
| 512 | think | 274.5 | 32.7 | 0.52 |
| 8192 | no-think | 318.5 | 30.3 | 24.6 |
| 8192 | think | 281.2 | 28.7 | 27.8 |
| 32768 | no-think | 187.7 | 26.1 | 172.6 |
| 32768 | think | 183.5 | 25.6 | 176.6 |
| 131072 | no-think | 90.6 | 18.4 | 1442.9 |
| 131072 | think | 94.6 | 18.5 | 1382.3 |

Prefill degrades super-linearly with context (within-request rate falls as KV grows:
~120 tok/s early at 1M → ~28 avg at 58%). mac3 reference: A@32k prefill 283, decode 24.8
(m41 prefill 0.66×, decode 1.05× at 32k). Thinking mode adds reasoning tokens even at
`reasoning_effort: none` (~98% of runs show reasoning_content) — kwarg behavior noted in
smoke; both modes recorded as configured.

## Track B — precision ladder (medians of 3; runtime per leg)

| quant | runtime | ctx 512 prefill / decode | ctx 32768 prefill / decode |
|---|---|---|---|
| Q4_K_XL | unsloth-mix | 481.9 / 32.6 | 187.0 / 26.0 |
| Q8_K_XL | unsloth-mix | 483.8 / 29.6 | 188.3 / 23.7 |
| IQ4_NL | mainline | 470.9 / 35.2 | 187.4 / 27.6 |
| BF16 | mainline | 496.2 / 26.5 | 187.9 / 20.6 |

All four quants ran on m41; mac3's BF16 landed Thu 01:57:44Z (phase-2, track B exit 0,
final verify PASS, zero blocked points — fleet-first full suite). **mac3 BF16 row
(mainline 418fe97, medians of 3, verified against banked records):** prefill 579.1 /
290.7, decode 25.62 / 23.13 (@512 / @32k), ttft 0.66 s / 112.3 s, math 0.6; file sha256
7074c976… cross-verified against the relay manifest (same identity as m41's BF16).
Prefill at 32k is compute-bound and quant-insensitive (m41 187–188 across all four;
mac3 within ~2%). Decode ranks IQ4_NL > Q4 > Q8 > BF16 on both Macs; the BF16 decode
penalty vs Q4 is ~8% on mac3, ~19–21% on m41, vs ~2× on Spark.
**BF16 prefill crossover — not reproduced on Metal; Spark's claimed crossover rests on
a cross-track number (flagged Thu ~02:10Z).** Both Macs show BF16 prefill marginally
faster at BOTH rungs (m41 496.2 vs 481.9 @512, 187.9 vs 187.0 @32k; mac3 579.1 vs
570.6, 290.7 vs 285.1 — all within ~3%). The Spark report's "520 vs 431 @512" compares
BF16 Track B (519.8) against Q4 **Track A** (431.4 / 432.1, 128-token prefill prompt);
the banked spark-1 **Track B** Q4 @512 median is 864.6 (runs 682.2 / 867.3 / 864.6).
Matched B-track spark-1: BF16 slower at both rungs (519.8 vs 864.6 @512; 221.4 vs
402.8 @32k) — no crossover in like-for-like data. The @32k side of the Spark claim
(221 vs 403) is matched-track and stands. Runtime confound noted: Q4 = unsloth-mix,
BF16 = mainline on all hosts. Flagged to mac3 lead Thu ~02:10Z; mac3 verified
independently and routed the correction to the Spark lead (Thu ~02:15Z). This report
carries only the matched-track statement.

## Track C — capability evals (thinking off, 30 items)

score 0.80 (code 1.00, math 0.70), reasoning_tokens 0. Record:
`C-m41-q4_k_xl--items-30-thinking-off.json`.

## Track D — needle-in-haystack ladder (Q4_K_XL, depth plan = Spark shape)

| ctx | depths | accuracy median | failure mode (raw fields) |
|---|---|---|---|
| 16384 | 0.1 / 0.5 / 0.9 | 1.0 / 1.0 / 1.0 | 9/9 correct |
| 65536 | 0.1 / 0.5 / 0.9 | 1.0 / 1.0 / 1.0 | one genuine miss per depth (no budget hits) |
| 131072 | 0.5 | 0.0 | acc [1,0,0]; found_in_reasoning 3/3; budget_hit 0/3 — code found in reasoning, emitted as content 1/3 |
| 262144 | 0.5 | 0.0 | acc [0,0,0]; finish_reason length 3/3; found_in_reasoning [T,F,T] |
| 1047040 | 0.5 | **PENDING** | attempt-4 retry in flight (M6 detached transport; see Repair log 6) |

**Cross-platform finding:** large-context needle failure is not Metal-specific. mac3
reproduces 131k median 0 (2/3 length-budget) and 262k [0,0,0]; mac3 1M (D4 record,
landed Thu 01:28Z): 0/3, finish_reason length 3/3, answers EMPTY 3/3 — reasoning
consumed the full budget; fact surfaced in reasoning 1/3. Spark reference: 262k
median 0 (1/3 correct), 1M median 0 (3/3 length) with degenerate repetition at the
8192 probe. Retrieval failure on CUDA and Metal both; not a budget artifact. A
completed accuracy-0 1M row is a valid, comparable data point; m41's row completes
the picture ~THU 19:45Z.

1M attempt evidence (decision 2026-11-22T06:05Z, owner card adaptive default): mac3 1M
run-1 live 63% @ 5h healthy; floor-10 per-run cap 29.2 h holds ≥1.5× margin. mac3's own
attempt 1 died at 89% on a stale-config race (process error, not feasibility).

**1M run-1 actuals (m41, attempt 3b; server log 1795375186, locked Mon 13:53Z):**
total 65,648 s = 18.24 h; prefill 65,013 s / 1,033,976 tok = 15.90 tok/s avg; decode
2048 tok in 636 s = 3.22 tok/s. mac3 run-1 same day: total 45,050 s = 12.51 h; prefill
23.16 tok/s avg; decode ~5.0 tok/s. Completion ratio m41/mac3 = **1.458** — the
mid-curve ratio (~1.26–1.30) drifted toward the bandwidth-bound tail as predicted.
Run 2 from-zero on both Macs: mac3 run-2 identical-pace to run-1 at equal token count
(no KV prefix reuse, no inter-run decay at 1M); m41 run-2 reached 79% (831,498 tok,
t = 43,593 s, 19.07 tok/s avg) before an upstream network event killed the transport
Tue 01:42Z (Repair log 6). Run-1 + run-2 chat text was lost with the point record
(needle.py writes per-point only after 3 runs); run-1 total-time line and the full
server log (run-1 complete + run-2 0–79% rate curve) preserved in `results/evidence/`.
**Attempt-4 retry (M6 detached transport) launched Tue 10:49:47Z, 1M point ×3 only:**
~18.2 h/run (flat estimate) → run 1 actual Wed 05:12Z; refined Wed 22:25Z from run-2
pace (95% @t=61,081s; each doc ~1.2% longer + ~1%/run per-token decay): run 2 ~THU
00:15Z, run 3 ~THU 19:30Z, point record **~THU 19:30–20Z**.
**Retry run-1 actuals (COMPLETE Wed ~05:12Z, server log 1795517387):** total 65,591.9 s
= 18.22 h; prefill 64,908.6 s / 1,033,976 tok = 15.93 tok/s avg; decode 2048 tok in
683.2 s = 3.00 tok/s; graphs reused 2038 / reallocs 2024. Vs attempt-3b run 1
(65,648 s): **0.086% total difference** — determinism holds under M6; the transport
fix did not move measurements. m41/mac3 D4-run-1 completion ratio = **1.456**
(attempt-3b pair: 1.458).
**Retry run-2 actuals (COMPLETE Thu ~00:14Z, task 2556):** total 67,690.1 s = 18.80 h;
prefill 67,047.9 s / 1,046,592 tok = 15.61 tok/s avg; decode 1,984 tok in 642.2 s =
3.09 tok/s (context-ceiling stop, see below). Run2/run1 total ratio 1.0320 (mac3: 1.0241); m41/mac3
run-2 ratio **1.467**. Vs attempt-3b run-2 partial curve at identical token positions
(798,730 / 800,778 / 831,498): retry **2.8–3.0% faster** (42,390 s vs 43,593 s @831,498)
— observation only, cause unconfirmed (machine state in the pre-incident window);
run 1 matched within 0.086%. Run 3 launched ~00:26Z (task 5054, fresh slot, 33,276 MiB
prompt state > 8 GiB cache limit).
**Run-2 stop reason (verified from server log):** context-ceiling truncation, not EOS —
`stop processing: n_tokens = 1048575, truncated = 1`; prompt 1,046,592 + decode 1,984 =
1,048,576 = exactly server_ctx (ctx + 1536). The run-2 document landed at the filler
target (ctx − 512), leaving 1,984 < 2,048 decode budget. Run 1's document landed ~12.6k
under target, so its full 2,048-token generation fit (truncated = 0). mac3 shows the
identical ceiling stop (eval counts 1,046,592 + 1,984 = 1,048,576 — same as m41).
Rate measurements unaffected.
**4-token prompt difference RESOLVED (Thu, verified in both hosts' banked logs):
symmetric log-line artifact, NOT a chat-path difference.** The final `progress = 1.00`
line under-reports the final `prompt eval time` count by exactly 4 tokens in EVERY run
on both hosts (m41 banked log: run 1 progress 1,033,972 vs eval 1,033,976; run 2
progress 1,046,588 vs eval 1,046,592; mac3 identical in all three). The earlier
cross-host delta came from comparing mac3's progress line against m41's eval line.
Authoritative eval counts: prompts IDENTICAL both hosts (1,033,976 / 1,046,592 /
1,046,592), decode IDENTICAL (2,048 / 1,984 / 1,984). All rates use eval lines — zero
measurement impact. The within-run 4-token offset is a server logging detail
(mechanism unverified, harmless). Supporting identity evidence retained: harness bytes,
seeds, GGUF, build string, binary hashes (stub/impl/common), template sha256, argv,
tokenize probes, all 22 build-fetch byte counts equal. Observation CLOSED in both
reports; post-campaign reconstruction optional, no longer needed for this item.
**mac3 D4 record LANDED Thu 01:28:16Z (exit 0, verify PASS — fleet-first 1M record):**
needle_accuracy 0/3, finish_reason length 3/3, answers EMPTY 3/3 — all decode went to
reasoning (8,686 / 8,036 / 8,326 chars); found_in_reasoning [F,F,T]; budget_hit only
run 1 (2,048 tok), runs 2–3 ceiling-stopped at 1,984. Run totals 45,057.6 / 46,143.1 /
46,122.0 s; run 3 prefill 45,765.0 s / 1,046,592 tok = 22.87 avg, decode 357.0 s /
1,984 tok = 5.55 tok/s; run3/run2 = 0.9995. Evidence:
server-mac3-1M-D4-complete-1795517085.log (1,901 lines). mac3 phase-2 BF16 auto-started
behind it (gates passed, self-staged quants deleted size-guarded, 375 GiB free).
m41 point record pending run 3 (~THU 19:30–20Z); the cross-platform 1M row completes
then. The optimistic 0.72× decay repeat seen at 262k does NOT apply at 1M.

## Track E — concurrency (Q4_K_XL; medians)

| parallel | server ctx | output tok/s | p50 latency s | failed workers |
|---|---|---|---|---|
| 1 | 8192 | 31.6 | 8.1 | 0 |
| 4 | 8192 | 71.4 | 14.3 | 0 |
| 16 | 8192 | 100.5 | 40.8 | 0 |
| 64 | 32320 | 264.1 | 62.0 | 0 |

8.4× aggregate throughput at 64 workers, zero failures.

## Track G — vision (8 items, mmproj on)

accuracy 0.62, budget_hits 0, empty_answers 0. Record:
`G-m41-q4_k_xl--items-8-mmproj-true.json`.

## Track H — soak (3 × 600 s windows)

decode 27.27 tok/s median, 256 requests, thermal_state nominal in all 3 windows
(`pmset -g therm`). GPU watts: honest null — no sudo on m41, powermetrics unavailable;
sustained tok/s + thermal state are the soak evidence.

## Track J — speculative decoding (dspark drafter, mix runtime)

Completed Tue 02:05:27Z (attempt 3b, after the D failure; 4 records × 3 runs each).
Drafter: `dspark-Inkling-2-Mini-Q8_0.gguf` (402 MB), staged + sha256-verified.

| ctx | speculate | prefill tok/s | decode tok/s | TTFT s | acceptance |
|---|---|---|---|---|---|
| 512 | false | 321.6 | 32.5 | 0.40 | — |
| 512 | true | 323.7 | 18.5 | 0.40 | 0.101 |
| 32768 | false | 188.7 | 26.1 | 171.6 | — |
| 32768 | true | 184.8 | 12.5 | 175.3 | 0.123 |

**Finding: speculation is a net LOSS with this drafter on the mix runtime.** Decode
falls to 0.57× at ctx 512 and 0.48× at 32k; prefill and TTFT unchanged (speculation
only touches decode). Acceptance is the server log's cumulative draft-acceptance
counter (one value per server instance, hence identical across the 3 runs of a point):
~10–12% — far too low to amortize draft overhead.

**Fleet-wide negative result (16 banked records, 6 hosts, medians of 3):** the dspark
Q8_0 drafter on the unsloth mix runtime slows decode at every measured ctx on every
host, Metal AND CUDA. Acceptance (server-log cumulative counter, one value per server
instance) sits at 0.067–0.135 everywhere — far below breakeven.

| host | backend | decode ratio @512 | @32k | acceptance @512 / @32k |
|---|---|---|---|---|
| m41 (M4 Max) | Metal | 0.57× | 0.48× | 0.101 / 0.123 |
| mac3 (M3 Ultra) | Metal | 0.58× | 0.56× | 0.071 / 0.105 |
| spark-1 (DGX Spark) | CUDA | 0.46× | 0.54× | 0.083 / 0.135 |
| rtxspark-1 (RTX Spark) | CUDA | 0.45× | 0.47× | 0.134 / 0.095 |
| forge (RTX 5090) | CUDA | 0.54× | 0.58× | 0.075 / 0.100 |
| titan (RTX 6000) | CUDA | 0.24× | 0.26× | 0.067 / 0.096 |

Ratio = speculate-true decode ÷ speculate-false decode (medians). Sources: each
project's `results/J-<host>-q4_k_xl--ctx-{512,32768}-…-speculate-{false,true}.json`
(read-only cross-check 2026-11-24). Prefill and TTFT unchanged by speculation on all
hosts. This is a drafter/runtime pairing property, not a Metal- or host-specific one.
m41 records: `J-m41-q4_k_xl--ctx-{512,32768}-…-speculate-{false,true}.json`.

## Tracks F / K — blocked (informational)

- F (loop depth): loop count baked into GGUF (`LLM_KV_LOOP_COUNT`); requires re-exported
  `custom_loop_ggufs` — not staged. Blocked record `F-m41-q4_k_xl.json` (Tue 02:05:27Z).
- K (two-node): single-host project; not applicable. Blocked record `K-m41-q4_k_xl.json`
  (Tue 02:05:27Z).

## Repair log (comparability preserved throughout)

1. Attempt-1 track A failed at ctx-131072: flat `health_timeout_s=900` < ~1100 s needed
   (128k prefill ~116 tok/s early estimate; measured median 90.6). Evidence preserved:
   `results/evidence/A-failed-2026-11-21T183832Z.json` + `fail-A-…log` (moved out of
   verify's top-level glob, never deleted).
2. 256 shipped shared harness fix: `prefill_tok_s_floor` + `_common.long_request_timeout
   = max(600, tokens/floor + 300)` used by A and D; `complete()` gained `timeout_s`.
   Re-synced byte-identical (176 tests). Config M4: floor 30 (Quick check SHIP).
3. Caps are PER RUN (needle.py:171–192), not per point. Config M5: floor 30→10 for 1M
   per-run patience (Quick check SHIP). Timeouts are caps only — no measured value
   changes; Spark default 150 untouched; mac3 also at floor 10.
4. Controlled stops at record boundaries (19:24Z after H; 23:15Z after D-16k) to adopt
   config changes — running processes hold cfg in memory. Stop markers:
   `results/stop-m41.json`, `results/stop-m41-2.json`.
5. verify.py semantics: no per-ctx coverage enforcement; blocked records informational;
   any top-level `state: failed` record fails the gate (hence the evidence move).
6. Network incident Tue 01:40–02:18Z: upstream transport event (not either Mac) killed
   the silent SSH request session carrying D-1M run 2 at 79% (rhttp exit 255, "Read from
   remote host … Operation timed out"); mac3's run 3 died in the same event. Suite
   continued: J/F/K completed, supervisor exited normally. Failed record preserved
   byte-intact → `results/evidence/D-failed-2026-11-24T014540Z.json` + full 1128-line
   server log (run-1 complete + run-2 partial rate curve); verify re-run PASS. Full
   record: `results/incident-m41-network-20261124.json`. Mitigations: client-side SSH
   keepalive (Host m41, mesh untouched) + **M6 harness fix from the 256 project**
   (rhttp requests with timeout_s > 600 run detached on-box: nohup wrapper, atomic
   exitcode, 30 s short-SSH polls, ~2 h SSH-failure tolerance, fetch-with-retries,
   cleanup only after both fetches succeed; 176 + 15 = 191 tests). Re-sync proven
   (shasum + diff identical excl configs; live 70 s detached smoke on m41 PASS 91.2 s).
   Attempt-4 retry launched Tue 10:49:47Z via `scripts/retry-1m-m41.sh` with
   `config-m41-1m-retry.json` (needle_plan = 1M point only; all else byte-identical;
   work box: grok-code writer after Cursor no-start, Quick check SHIP, 191 tests +
   dry-runs + semantic diff). Banked points untouched; same seeds reproduce the same
   documents.

## Finish gate (pending)

`python3 harness/verify.py results/ --config harness/config-m41.json` → PASS required
(approved tracks A B C D E G H; ≥3 runs + median + evidence each; F/J/K informational).
Interim verify PASS at 2026-11-22T11:59Z with D partial; post-incident verify PASS Tue
~02:30Z after failed-record preservation (`logs/verify-m41-postincident.log`). Final
verify runs automatically at attempt-4 exit (`results/verify-m41.exit`,
`logs/verify-m41.log`), then claim-audit and Done card.

## Campaign log anchors

- `HANDOFF.md` — live tracker, decisions, hard stops.
- `logs/suite-m41.log` (attempt 1), `logs/suite-m41-resume.log` (attempt 2),
  `logs/suite-m41-resume2.log` (attempt 3/3b), `logs/suite-m41-retry1m.log` (attempt 4).
- `results/incident-m41-network-20261124.json` — network incident record;
  `results/evidence/` — preserved failed records, partial server logs, attempt markers.
- Forgejo memory: topic `inkling2-m41`, PRs #269 #271 #276 #281 #284 #287 #292 …
  #384 #385 #386 #390 (incident, retry prep, M6 re-sync + launch).