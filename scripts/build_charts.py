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
        subtitle=("Median of 3 runs, thinking off, plain decode (no speculative decoding) on 4 NVIDIA Blackwell machines.",),
        footnotes=(
            "RTX 5090 also completes 256K all-GPU (f16 KV) at 46.4 tok/s. Strata SQ-20 and NVFP4 measured on single RTX PRO 6000 (600 W).",
        ),
    )
    return line_panel(caption, panel, theme)


def build_02_dspark_reversal(theme: Theme) -> str:
    native_panel = BarPanel(
        title="Native engines (DSpark ON — 2.0× to 4.1× speedup)",
        rows=[
            BarRow("RTX PRO 6000 · Strata native*", 262.9, "262.9 tok/s", "aqua", "2.04× vs 128.7"),
            BarRow("RTX Spark · vLLM (NVFP4)", 183.5, "183.5 tok/s", "blue", "4.11× vs 44.6"),
            BarRow("DGX Spark · vLLM (NVFP4)", 142.01, "142.0 tok/s", "blue", "3.81× vs 37.3"),
            BarRow("RTX Spark · SGLang (NVFP4)", 140.3, "140.3 tok/s", "violet", "3.14× vs 44.7"),
            BarRow("DGX Spark · SGLang (NVFP4)", 119.69, "119.7 tok/s", "violet", "3.23× vs 37.1"),
        ],
    )
    llama_panel = BarPanel(
        title="llama.cpp GGUF (DSpark ON — 0.45× to 0.58× slowdown)",
        rows=[
            BarRow("RTX PRO 6000 · llama.cpp", 50.21, "50.2 tok/s", "red", "0.58× vs 87.1 (6% acc)"),
            BarRow("RTX 5090 · llama.cpp", 44.4, "44.4 tok/s", "red", "0.54× vs 82.7 (7.5% acc)"),
            BarRow("DGX Spark · llama.cpp", 17.95, "18.0 tok/s", "orange", "0.46× vs 39.0 (8.3% acc)"),
            BarRow("RTX Spark · llama.cpp", 17.06, "17.1 tok/s", "orange", "0.45× vs 37.8 (13.4% acc)"),
        ],
    )
    caption = Caption(
        title="The DSpark speculative decoding reversal: native engines vs llama.cpp",
        subtitle=("Decode throughput at ctx 512 (median of 3). Same Inkling-2-Mini model; opposite outcome by engine.",),
        footnotes=(
            "*Strata native bench (BEAM-W4 TREE-6, 256/256 bit-exact parity); Strata v0.2.19 HTTP serve API does not engage DSpark (155.5 vs 155.5).",
            "Shared horizontal scale across both panels so bar lengths are directly comparable.",
        ),
    )
    return bar_panels(caption, [native_panel, llama_panel], theme, columns=1, axis_label="Decode throughput with DSpark enabled (tok/s)", shared_scale=True)


def build_03_concurrency(theme: Theme) -> str:
    p64_panel = BarPanel(
        title="Aggregate output at 64 parallel streams (0 failed workers)",
        rows=[
            BarRow("RTX PRO 6000 · Strata SQ-20", 1046.6, "1,046.6 tok/s", "aqua", "5.4× vs 1-stream"),
            BarRow("RTX PRO 6000 · llama.cpp IQ4_NL", 978.0, "978.0 tok/s", "blue", "10.6× vs 1-stream"),
            BarRow("RTX Spark · SGLang (NVFP4)", 922.9, "922.9 tok/s", "violet", "20.7× vs 1-stream"),
            BarRow("RTX PRO 6000 · llama.cpp Q4_K_XL", 906.77, "906.8 tok/s", "blue", "9.8× vs 1-stream"),
            BarRow("DGX Spark · SGLang (NVFP4)", 842.02, "842.0 tok/s", "violet", "23.3× vs 1-stream"),
            BarRow("RTX Spark · vLLM (NVFP4)", 728.6, "728.6 tok/s", "yellow", "21.7× vs 1-stream"),
            BarRow("DGX Spark · vLLM (NVFP4)", 645.0, "645.0 tok/s", "yellow", "21.8× vs 1-stream"),
            BarRow("RTX 5090 · llama.cpp Q4_K_XL", 601.7, "601.7 tok/s", "orange", "7.0× vs 1-stream"),
            BarRow("RTX Spark · llama.cpp Q4_K_XL", 374.5, "374.5 tok/s", "orange", "10.1× vs 1-stream"),
            BarRow("DGX Spark · llama.cpp Q4_K_XL", 368.75, "368.8 tok/s", "orange", "10.0× vs 1-stream"),
            BarRow("RTX PRO 6000 · Strata NVFP4 (#6520)", 140.3, "140.3 tok/s", "red", "serializes (1.0×)"),
            BarRow("RTX PRO 6000 · Strata GGUF (#6642)", 127.0, "127.0 tok/s", "red", "1.4× vs 1-stream"),
        ],
    )
    caption = Caption(
        title="Concurrency scaling at 64 parallel streams (on-box HTTP load)",
        subtitle=("Aggregate output tok/s across 64 simultaneous 256-token requests. Every engine finished with zero failed workers.",),
        footnotes=(
            "RTX Spark running SGLang (922.9 tok/s) matches single-card RTX PRO 6000 running llama.cpp Q4_K_XL (906.8 tok/s).",
            "Strata #6520 NVFP4 has no continuous batch slots and serializes concurrent requests (~140 tok/s flat from 1 to 64).",
        ),
    )
    return bar_panels(caption, [p64_panel], theme, columns=1, axis_label="Aggregate output throughput at 64 concurrent streams (tok/s)")


def build_04_power_knee(theme: Theme) -> str:
    speed_panel = BarPanel(
        title="Decode @128K (thinking on)",
        rows=[
            BarRow("350 W cap (throttled)", 23.95, "24.0 tok/s", "red", "baseline"),
            BarRow("450 W cap (sweet spot)", 63.92, "63.9 tok/s", "aqua", "+167%"),
            BarRow("600 W cap (card max)", 63.89, "63.9 tok/s", "blue", "+167%"),
        ],
    )
    soak_panel = BarPanel(
        title="30-minute soak decode",
        rows=[
            BarRow("350 W cap (throttled)", 39.54, "39.5 tok/s", "red", "15% fade"),
            BarRow("450 W cap (sweet spot)", 65.72, "65.7 tok/s", "aqua", "flat, +66%"),
            BarRow("600 W cap (card max)", 66.14, "66.1 tok/s", "blue", "flat, +67%"),
        ],
    )
    watts_panel = BarPanel(
        title="Mean soak power draw (lower is better)",
        rows=[
            BarRow("350 W cap (throttled)", 327.0, "327 W", "red", "84 °C max"),
            BarRow("450 W cap (sweet spot)", 221.0, "221 W", "aqua", "67 °C max"),
            BarRow("600 W cap (card max)", 221.0, "221 W", "blue", "66 °C max"),
        ],
    )
    eff_panel = BarPanel(
        title="Sustained efficiency (tok/W)",
        rows=[
            BarRow("350 W cap (throttled)", 0.121, "0.121 tok/W", "red", "pinned clocks"),
            BarRow("450 W cap (sweet spot)", 0.299, "0.299 tok/W", "aqua", "2.47× better"),
            BarRow("600 W cap (card max)", 0.299, "0.299 tok/W", "blue", "2.47× better"),
        ],
    )
    caption = Caption(
        title="RTX PRO 6000 Blackwell power ladder: 350 W vs 450 W vs 600 W",
        subtitle=("Same card, same weights, same harness. Escaping 350 W removes SW Power Cap throttling and cuts both watts and heat.",),
        footnotes=(
            "At 350 W, SW Power Cap throttles clocks (~1800 MHz), raising duty cycle to 327 W mean / 84 °C.",
            "At 450 W and 600 W, the workload peaks at ~230 W (not cap-bound) and finishes faster. 450 W buys 100% of 600 W performance.",
        ),
    )
    return bar_panels(caption, [speed_panel, soak_panel, watts_panel, eff_panel], theme, columns=2)


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
            BarRow("Strata SQ-20", 94.98, "95.0 tok/s", "aqua", "1.48× base"),
            BarRow("llama.cpp Q4_K_XL", 63.96, "64.0 tok/s", "blue", "baseline"),
            BarRow("llama.cpp IQ4_NL", 63.58, "63.6 tok/s", "blue", "same-wt ref"),
            BarRow("Strata GGUF (#6642)", 57.13, "57.1 tok/s", "orange", "0.90× llama"),
        ],
    )
    eval_panel = BarPanel(
        title="Combined eval score (20 math + 10 code)",
        rows=[
            BarRow("Strata NVFP4 (#6520)", 86.7, "0.867", "violet", "16/20 math"),
            BarRow("Strata SQ-20", 80.0, "0.800", "aqua", "14/20 math"),
            BarRow("llama.cpp Q4_K_XL", 80.0, "0.800", "blue", "14/20 math"),
            BarRow("llama.cpp IQ4_NL", 73.3, "0.733", "blue", "12/20 math"),
            BarRow("Strata GGUF (#6642)", 53.3, "0.533", "red", "6/20 math"),
        ],
    )
    eff_panel = BarPanel(
        title="30-minute soak efficiency (tok/W)",
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
        subtitle=("Comparing llama.cpp baseline and IQ4_NL against Strata SQ-20, PR #6520 NVFP4, and PR #6642 GGUF.",),
        footnotes=(
            "Code eval = 10/10 on all five arms. Strata SQ-20 OOMs on the 1M needle prompt (267K prefill); llama.cpp completes 1M.",
            "Strata #6642 GGUF and llama.cpp IQ4_NL ran the exact same .gguf file for a controlled head-to-head.",
        ),
    )
    return bar_panels(caption, [single_panel, long_panel, eval_panel, eff_panel], theme, columns=2)


def build_06_1m_breakthrough(theme: Theme) -> str:
    panel_1m = BarPanel(
        title="1,047,040-token needle retrieval (3 seeds, depth 0.5)",
        rows=[
            BarRow("DGX Spark · vLLM max-bf16 + prefill", 100.0, "3/3 (100%)", "aqua", "9 output tok, clean stop"),
            BarRow("DGX Spark · vLLM max-fp8 (chat)", 2.0, "0/3 (0%)", "red", "budget hit; 1/3 in reasoning"),
            BarRow("DGX Spark · SGLang max (chat)", 2.0, "0/3 (0%)", "red", "budget hit; 2/3 in reasoning"),
            BarRow("RTX Spark · vLLM & SGLang (chat)", 2.0, "0/3 (0%)", "red", "matches DGX run-for-run"),
            BarRow("DGX / RTX / 6000 · llama.cpp (chat)", 2.0, "0/3 (0%)", "orange", "degenerate loop at 8k probe"),
            BarRow("RTX PRO 6000 · Strata SQ-20 (chat)", 0.0, "BLOCKED", "yellow", "CUDA OOM at 267K prefill"),
        ],
    )
    caption = Caption(
        title="1M-context needle retrieval: standard chat vs assistant prefill",
        subtitle=("Standard chat fails 0/3 on every engine because the template ignores reasoning_effort=none (~98.4% reasoning tokens).",),
        footnotes=(
            "Fix: chat-assistant-prefill + continue_final_message bypasses the reasoning loop → 3/3 exact hits (XJ9-4782, XJ9-6498, XJ9-7378).",
            "Wall time on DGX Spark vLLM max-bf16: 1,515–1,564 s (~25.5 min/run). Zero-hit rows shown as stubs for visibility.",
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
