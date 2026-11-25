#!/usr/bin/env python3
"""Build all Inkling-2-Mini SVG charts (dark + light themes) from data/benchmarks.json.

Standard library only — deterministic, byte-identical output matching omlx-fleet-bench.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from charts import BarPanel, BarRow, Caption, LinePanel, LinePoint, LineSeries, bar_panels, line_panel
from svg import THEMES, Theme

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
CHARTS = ROOT / "charts"
CHARTS.mkdir(exist_ok=True)


def write_themed(stem: str, builder) -> None:
    for theme_name, theme in THEMES.items():
        svg = builder(theme)
        out = CHARTS / f"{stem}-{theme_name}.svg"
        out.write_text(svg, encoding="utf-8")
        print(f"Wrote {out.name}")


def build_01_speed_ladder(theme: Theme) -> str:
    lookup: Dict[Tuple[str, str, int], float] = {
        (r["machine_id"], r["engine"], r["ctx"]): r["decode_tps"] for r in DATA["speed_ladder"]
    }
    ctxs = [(512, "512"), (8192, "8K"), (32768, "32K"), (131072, "128K")]
    specs = [
        ("RTX PRO 6000 · Strata SQ-20", "rtx-6000", "Strata SQ-20", "blue"),
        ("RTX PRO 6000 · Strata NVFP4", "rtx-6000", "Strata NVFP4 (#6520)", "violet"),
        ("RTX PRO 6000 · llama.cpp", "rtx-6000", "llama.cpp (600W)", "aqua"),
        ("RTX 5090 · llama.cpp", "rtx-5090", "llama.cpp", "orange"),
        ("RTX Spark · vLLM", "rtx-spark", "vLLM", "yellow"),
        ("DGX Spark · vLLM", "dgx-spark", "vLLM", "magenta"),
        ("DGX Spark · llama.cpp", "dgx-spark", "llama.cpp", "red"),
    ]
    series = [
        LineSeries(
            label=label,
            series=slot,
            points=[
                LinePoint(x=c, y=lookup[(mid, eng, c)], x_text=clabel, y_text=f"{lookup[(mid, eng, c)]:.1f}")
                for c, clabel in ctxs
            ],
        )
        for label, mid, eng, slot in specs
    ]
    panel = LinePanel(
        title="Decode throughput across context",
        x_min=512,
        x_max=131072,
        y_min=0,
        y_max=200,
        x_ticks=[(512, "512"), (8192, "8K"), (32768, "32K"), (131072, "128K")],
        y_ticks=[(0, "0"), (50, "50"), (100, "100"), (150, "150"), (200, "200")],
        series=series,
        x_label="Prompt context length (tokens, log scale)",
        y_label="Decode throughput (tok/s)",
        log_x=True,
        log_y=False,
    )
    caption = Caption(
        title="Single-stream decode speed across context lengths (512 to 128K)",
        subtitle=("Median of 3 runs, plain decode (no speculative decoding). Ecosystem stacks: GGUF Q4_K_XL, native NVFP4, Strata SQ-20.",),
        footnotes=(
            "vLLM and SGLang were not tested on RTX PRO 6000 or RTX 5090. RTX 5090 also completes 256K all-GPU (f16 KV) at 46.4 tok/s.",
            "On llama.cpp/vLLM/SGLang, reasoning_effort=none still emits ~98.4% reasoning tokens; Strata hides reasoning_content.",
        ),
    )
    return line_panel(caption, panel, theme)


def build_02_dspark_reversal(theme: Theme) -> str:
    native_panel = BarPanel(
        title="Native engines (DSpark ON — upper bound on synthetic NATO completions*)",
        rows=[
            BarRow("RTX PRO 6000 · Strata native†", 262.9, "262.9 tok/s", "aqua", "2.04× vs 128.7 (exact)"),
            BarRow("RTX Spark · vLLM (NVFP4)*", 183.5, "183.5 tok/s", "blue", "4.11× (range 85–208)"),
            BarRow("DGX Spark · vLLM (NVFP4)*", 142.01, "142.0 tok/s", "blue", "3.81× (range 68–160)"),
            BarRow("RTX Spark · SGLang (NVFP4)*", 140.3, "140.3 tok/s", "violet", "3.14× (range 92–185)"),
            BarRow("DGX Spark · SGLang (NVFP4)*", 119.69, "119.7 tok/s", "violet", "3.23× (range 73–148)"),
        ],
    )
    llama_panel = BarPanel(
        title="llama.cpp Unsloth mix GGUF drafter (DSpark ON — net slowdown)",
        rows=[
            BarRow("RTX PRO 6000 · llama.cpp", 50.21, "50.2 tok/s", "red", "0.58× vs 87.1 (6% acc)"),
            BarRow("RTX 5090 · llama.cpp", 44.4, "44.4 tok/s", "red", "0.54× vs 82.7 (7.5% acc)"),
            BarRow("DGX Spark · llama.cpp", 17.95, "18.0 tok/s", "orange", "0.46× vs 39.0 (8.3% acc)"),
            BarRow("RTX Spark · llama.cpp", 17.06, "17.1 tok/s", "orange", "0.45× vs 37.8 (13.4% acc)"),
        ],
    )
    caption = Caption(
        title="DSpark speculative decoding: native engines vs early llama.cpp GGUF drafter",
        subtitle=("Decode throughput at ctx 512 (median of 3). Native engines use official mtp.safetensors; llama.cpp uses Unsloth b12588-mix Q8_0 drafter.",),
        footnotes=(
            "*Caveat: vLLM/SGLang Track J ran raw NATO-filler completions that looped on 1-word tokens, inflating acceptance (upper bound, high variance).",
            "†Strata native bench (BEAM-W4 TREE-6) is bit-exact (256/256) on non-degenerate text; Strata v0.2.19 HTTP serve API does not engage DSpark (1.00×).",
        ),
    )
    return bar_panels(caption, [native_panel, llama_panel], theme, columns=1, axis_label="Decode throughput with DSpark enabled (tok/s)", shared_scale=True)


def build_03_concurrency(theme: Theme) -> str:
    p64_panel = BarPanel(
        title="Aggregate output at 64 parallel streams (identical short prompt, 0 failures)",
        rows=[
            BarRow("RTX PRO 6000 · Strata SQ-20", 1046.6, "1,046.6 tok/s", "aqua", "5.4× vs 1-stream"),
            BarRow("RTX PRO 6000 · llama.cpp IQ4_NL", 977.7, "977.7 tok/s", "blue", "10.6× vs 1-stream"),
            BarRow("RTX Spark · SGLang (NVFP4)*", 922.9, "922.9 tok/s", "violet", "20.7× (warm radix)"),
            BarRow("RTX PRO 6000 · llama.cpp Q4_K_XL", 906.77, "906.8 tok/s", "blue", "9.8× vs 1-stream"),
            BarRow("DGX Spark · SGLang (NVFP4)*", 842.02, "842.0 tok/s", "violet", "23.3× (warm radix)"),
            BarRow("RTX Spark · vLLM (NVFP4)*", 728.6, "728.6 tok/s", "yellow", "21.7× (prefix cache)"),
            BarRow("DGX Spark · vLLM (NVFP4)*", 645.0, "645.0 tok/s", "yellow", "21.8× (prefix cache)"),
            BarRow("RTX 5090 · llama.cpp Q4_K_XL", 601.7, "601.7 tok/s", "orange", "7.0× vs 1-stream"),
            BarRow("RTX Spark · llama.cpp Q4_K_XL", 374.5, "374.5 tok/s", "orange", "10.1× vs 1-stream"),
            BarRow("DGX Spark · llama.cpp Q4_K_XL", 368.75, "368.8 tok/s", "orange", "10.0× vs 1-stream"),
            BarRow("RTX PRO 6000 · Strata NVFP4 (#6520)", 140.3, "140.3 tok/s", "red", "serializes (1.0×)"),
            BarRow("RTX PRO 6000 · Strata GGUF (#6642)†", 126.6, "126.6 tok/s", "red", "early-EOS (~52 tok)"),
        ],
    )
    caption = Caption(
        title="Concurrency scaling at 64 parallel streams (engine comparison, not same-engine hardware)",
        subtitle=("Aggregate output tok/s across 64 simultaneous 256-token requests. On identical llama.cpp Q4_K_XL, PRO 6000 is 2.42× RTX Spark.",),
        footnotes=(
            "*vLLM/SGLang were NOT tested on PRO 6000 or 5090. All 64 workers sent the same 512-char prompt with prefix/radix caching active (warm median).",
            "†Strata #6642 GGUF truncated early (~52 tok/req instead of 256), making its 126.6 tok/s non-comparable. #6520 NVFP4 has no batch slots.",
        ),
    )
    return bar_panels(caption, [p64_panel], theme, columns=1, axis_label="Aggregate output throughput at 64 concurrent streams (tok/s)")


def build_04_power_knee(theme: Theme) -> str:
    speed_panel = BarPanel(
        title="Decode @512 and @128K (450 W vs 600 W)",
        rows=[
            BarRow("450 W · decode @512", 87.26, "87.3 tok/s", "aqua", "matches 600 W"),
            BarRow("600 W · decode @512", 86.76, "86.8 tok/s", "blue", "card max"),
            BarRow("450 W · decode @128K", 63.92, "63.9 tok/s", "aqua", "matches 600 W"),
            BarRow("600 W · decode @128K", 63.89, "63.9 tok/s", "blue", "card max"),
        ],
    )
    soak_panel = BarPanel(
        title="30-minute serial soak (450 W vs 600 W)",
        rows=[
            BarRow("450 W · soak decode", 65.72, "65.7 tok/s", "aqua", "221 W mean, 67 °C"),
            BarRow("600 W · soak decode", 66.14, "66.1 tok/s", "blue", "222 W mean, 66 °C"),
            BarRow("350 W* · day-1 baseline", 39.54, "39.5 tok/s", "red", "327 W / 84 °C (suspect)"),
        ],
    )
    caption = Caption(
        title="RTX PRO 6000 Blackwell power envelope: 450 W matches 600 W",
        subtitle=("Single-stream llama.cpp Q4_K_XL naturally draws ~221 W mean / ~230 W peak during serial soak, so 450 W and 600 W are identical.",),
        footnotes=(
            "*Audit note: the day-1 350 W run drew 327 W mean / 84 °C while running slower (physically inconsistent with a pure power cap; likely co-resident load).",
            "Track H runs serial 64-token requests with inter-request nvidia-smi polling (not a saturated TDP stress test); per-point SM clocks were not logged.",
        ),
    )
    return bar_panels(caption, [speed_panel, soak_panel], theme, columns=2)


def build_05_rtx6000_engines(theme: Theme) -> str:
    single_panel = BarPanel(
        title="Single-stream decode @512",
        rows=[
            BarRow("Strata SQ-20", 178.6, "178.6 tok/s", "aqua", "2.06× base"),
            BarRow("Strata NVFP4 (#6520)", 163.06, "163.1 tok/s", "violet", "1.88× base"),
            BarRow("llama.cpp IQ4_NL", 95.37, "95.4 tok/s", "blue", "same-wt ref"),
            BarRow("Strata GGUF (#6642)", 93.64, "93.6 tok/s", "orange", "0.98× llama"),
            BarRow("llama.cpp Q4_K_XL", 86.76, "86.8 tok/s", "blue", "baseline"),
        ],
    )
    long_panel = BarPanel(
        title="Single-stream decode @128K",
        rows=[
            BarRow("Strata NVFP4 (#6520)", 96.02, "96.0 tok/s", "violet", "1.50× base"),
            BarRow("Strata SQ-20*", 94.98, "95.0 tok/s", "aqua", "run-2 dip 41.7"),
            BarRow("llama.cpp Q4_K_XL", 63.96, "64.0 tok/s", "blue", "baseline"),
            BarRow("llama.cpp IQ4_NL", 63.58, "63.6 tok/s", "blue", "same-wt ref"),
            BarRow("Strata GGUF (#6642)", 57.13, "57.1 tok/s", "orange", "0.90× llama"),
        ],
    )
    eval_panel = BarPanel(
        title="Sanity evals (20 math + 10 code, default temp)",
        rows=[
            BarRow("Strata NVFP4 (#6520)", 86.7, "0.867", "violet", "range .80–.90"),
            BarRow("Strata SQ-20", 80.0, "0.800", "aqua", "14/20 math"),
            BarRow("llama.cpp Q4_K_XL", 80.0, "0.800", "blue", "14/20 math"),
            BarRow("llama.cpp IQ4_NL", 73.3, "0.733", "blue", "12/20 math"),
            BarRow("Strata GGUF (#6642)", 53.3, "0.533", "red", "early-EOS bug"),
        ],
    )
    eff_panel = BarPanel(
        title="30-minute serial soak (effective tok/W)",
        rows=[
            BarRow("llama.cpp Q4_K_XL", 0.299, "0.299", "blue", "222 W / 66 °C"),
            BarRow("llama.cpp IQ4_NL", 0.293, "0.293", "blue", "238 W / 69 °C"),
            BarRow("Strata SQ-20", 0.278, "0.278", "aqua", "345 W / 82 °C"),
            BarRow("Strata NVFP4 (#6520)", 0.262, "0.262", "violet", "350 W / 80 °C"),
            BarRow("Strata GGUF (#6642)", 0.210, "0.210", "red", "299 W / 74 °C"),
        ],
    )
    caption = Caption(
        title="RTX PRO 6000 (600 W): engine and quant trade-offs",
        subtitle=("Comparing llama.cpp baseline and IQ4_NL against Strata SQ-20, PR #6520 NVFP4 (CUDA 12.8 override), and PR #6642 GGUF.",),
        footnotes=(
            "*Strata SQ-20 @128K runs were [95.0, 105.0, 41.7]; SQ-20 also OOMs at 267K on the 1M needle prompt. #6642 vs IQ4_NL is the only same-weight pair.",
            "Track C evals (n=20 math, n=10 code) omitted temperature=0 and serve as regression sanity checks, not leaderboard rankings.",
        ),
    )
    return bar_panels(caption, [single_panel, long_panel, eval_panel, eff_panel], theme, columns=2)


def build_06_1m_breakthrough(theme: Theme) -> str:
    panel_1m = BarPanel(
        title="1,047,040-token needle retrieval (depth 0.5, synthetic NATO haystack)",
        rows=[
            BarRow("DGX Spark · vLLM max-bf16 + prefill*", 100.0, "3/3 (100%)", "aqua", "9 tok, ~25.5 min/run"),
            BarRow("DGX Spark · vLLM max-fp8 (std chat)", 2.0, "0/3 (0%)", "red", "budget hit; 1/3 in reasoning"),
            BarRow("DGX Spark · SGLang max (std chat)", 2.0, "0/3 (0%)", "red", "budget hit; 2/3 in reasoning"),
            BarRow("RTX Spark · vLLM & SGLang (std chat)", 2.0, "0/3 (0%)", "red", "matches DGX run-for-run"),
            BarRow("DGX / RTX / 6000 · llama.cpp (chat)", 2.0, "0/3 (0%)", "orange", "degenerate loop at 8k probe"),
            BarRow("RTX PRO 6000 · Strata SQ-20 (chat)", 0.0, "BLOCKED", "yellow", "CUDA OOM at 267K prefill"),
        ],
    )
    caption = Caption(
        title="1M-context diagnostic: standard chat fails 0/3; assistant prefill proves KV retention",
        subtitle=("Standard chat fails because the template ignores reasoning_effort=none (~98.4% reasoning tokens) and loops to budget exhaustion.",),
        footnotes=(
            "*Diagnostic scope: n=3, depth 0.5 only, single NATO haystack, forced 4-word assistant prefix ('The hidden code is') on vLLM max-bf16.",
            "Proves the 1M KV attention state retains the needle (not general 1M reasoning). Raw forced-prefix Arm B hallucinated 3X7-5778.",
        ),
    )
    return bar_panels(caption, [panel_1m], theme, columns=1, axis_label="Exact needle code retrieved in final answer (%)")


def main() -> None:
    write_themed("01-single-stream-decode", build_01_speed_ladder)
    write_themed("02-dspark-reversal", build_02_dspark_reversal)
    write_themed("03-concurrency-scaling", build_03_concurrency)
    write_themed("04-rtx6000-power-knee", build_04_power_knee)
    write_themed("05-rtx6000-engines", build_05_rtx6000_engines)
    write_themed("06-1m-breakthrough", build_06_1m_breakthrough)


if __name__ == "__main__":
    main()
