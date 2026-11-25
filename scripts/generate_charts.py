#!/usr/bin/env python3
"""Generate publication-ready, X/Twitter-optimized PNG charts from data/benchmarks.json.

Run with:
    uv run --with matplotlib --with numpy python3 scripts/generate_charts.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
CHARTS = ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

# Dark editorial palette (high contrast for X/Twitter & GitHub dark mode)
BG = "#0B0F17"
PANEL = "#121826"
GRID = "#1F2937"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
ACCENT_CYAN = "#22D3EE"
ACCENT_EMERALD = "#34D399"
ACCENT_AMBER = "#FBBF24"
ACCENT_CORAL = "#FB7185"
ACCENT_VIOLET = "#A78BFA"
ACCENT_BLUE = "#60A5FA"
ACCENT_ORANGE = "#FB923C"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Inter", "Helvetica", "Arial"],
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def add_header_footer(fig, title: str, subtitle: str, footnote: str):
    fig.text(0.05, 0.94, title, fontsize=22, fontweight="bold", color=TEXT)
    fig.text(0.05, 0.895, subtitle, fontsize=12.5, color=MUTED)
    fig.text(0.05, 0.036, footnote, fontsize=9.2, color=MUTED)
    fig.text(0.95, 0.012, "Inkling-2-Mini NVIDIA Benchmark Suite  •  Median of 3 Runs  •  Verify PASS",
             fontsize=9.0, color="#64748B", ha="right")


def save(fig, filename: str):
    out = CHARTS / filename
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out.name}")


def chart_01_single_stream():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.77, top=0.84, bottom=0.12)

    ctxs = [512, 8192, 32768, 131072]
    x = np.arange(len(ctxs))
    ctx_labels = ["512", "8K", "32K", "128K"]

    series = [
        ("RTX PRO 6000 • Strata SQ-20", "rtx-6000", "Strata SQ-20", ACCENT_CYAN, "o", 3.2, 6, -12),
        ("RTX PRO 6000 • Strata NVFP4 (#6520)", "rtx-6000", "Strata NVFP4 (#6520)", ACCENT_VIOLET, "s", 3.0, 6, 5),
        ("RTX PRO 6000 • llama.cpp (600W)", "rtx-6000", "llama.cpp (600W)", ACCENT_BLUE, "^", 2.5, 6, -3),
        ("RTX 5090 • llama.cpp", "rtx-5090", "llama.cpp", ACCENT_ORANGE, "D", 2.5, -14, -3),
        ("RTX Spark • vLLM (NVFP4)", "rtx-spark", "vLLM", ACCENT_EMERALD, "o", 2.6, 6, 3),
        ("DGX Spark • vLLM (NVFP4)", "dgx-spark", "vLLM", ACCENT_AMBER, "s", 2.4, 5, -2),
        ("DGX Spark • llama.cpp (Q4_K_XL)", "dgx-spark", "llama.cpp", MUTED, "--", 1.8, -14, -12),
    ]

    lookup = {(r["machine_id"], r["engine"], r["ctx"]): r["decode_tps"] for r in DATA["speed_ladder"]}

    for label, mid, eng, color, marker, lw, dy0, dy1 in series:
        ys = [lookup[(mid, eng, c)] for c in ctxs]
        if marker == "--":
            ax.plot(x, ys, linestyle="--", marker="x", color=color, linewidth=lw, markersize=7, label=label)
        else:
            ax.plot(x, ys, marker=marker, color=color, linewidth=lw, markersize=8, label=label)
        # Annotate start and end
        ax.annotate(f"{ys[0]:.1f}", (x[0], ys[0]), textcoords="offset points", xytext=(-18, dy0),
                    fontsize=9.5, fontweight="bold", color=color)
        ax.annotate(f"{ys[-1]:.1f}", (x[-1], ys[-1]), textcoords="offset points", xytext=(8, dy1),
                    fontsize=9.5, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(ctx_labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("Prompt Context Length (tokens)", fontsize=12, labelpad=10)
    ax.set_ylabel("Decode Throughput (tok/s)", fontsize=12, labelpad=10)
    ax.set_ylim(0, 200)
    ax.grid(True, axis="y", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True,
              facecolor=PANEL, edgecolor=GRID, fontsize=10.5, labelspacing=0.9)

    add_header_footer(
        fig,
        "Inkling-2-Mini: Single-Stream Decode Speed Across Context (512 → 128K)",
        "4 NVIDIA Blackwell machines • Thinking off • Plain decode (no speculative decoding)",
        "Note: RTX 5090 also completes 256K all-GPU at 46.4 tok/s. Strata SQ-20 & NVFP4 run on single RTX PRO 6000 (600W)."
    )
    save(fig, "01_single_stream_decode.png")


def chart_02_dspark_reversal():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.22, right=0.93, top=0.83, bottom=0.12)

    rows = [
        ("RTX PRO 6000 • Strata Native*", 128.7, 262.9, "2.04×"),
        ("RTX Spark • vLLM (NVFP4)", 44.6, 183.5, "4.11×"),
        ("DGX Spark • vLLM (NVFP4)", 37.29, 142.01, "3.81×"),
        ("RTX Spark • SGLang (NVFP4)", 44.7, 140.3, "3.14×"),
        ("DGX Spark • SGLang (NVFP4)", 37.10, 119.69, "3.23×"),
        ("RTX PRO 6000 • llama.cpp", 87.13, 50.21, "0.58×"),
        ("RTX 5090 • llama.cpp", 82.7, 44.4, "0.54×"),
        ("DGX Spark • llama.cpp", 38.97, 17.95, "0.46×"),
        ("RTX Spark • llama.cpp", 37.78, 17.06, "0.45×"),
    ]
    rows = rows[::-1]
    y = np.arange(len(rows))
    h = 0.36

    off_vals = [r[1] for r in rows]
    on_vals = [r[2] for r in rows]
    labels = [r[0] for r in rows]
    ratios = [r[3] for r in rows]
    on_colors = [ACCENT_CORAL if on < off else ACCENT_EMERALD for off, on in zip(off_vals, on_vals)]

    ax.barh(y - h / 2, off_vals, height=h, color="#475569", label="Plain Decode (DSpark OFF)")
    bars_on = ax.barh(y + h / 2, on_vals, height=h, color=on_colors)

    for i, (off, on, ratio) in enumerate(zip(off_vals, on_vals, ratios)):
        c = ACCENT_CORAL if on < off else ACCENT_EMERALD
        ax.text(max(off, on) + 5, y[i], f"{on:.1f} tok/s  ({ratio})",
                va="center", fontsize=11, fontweight="bold", color=c)

    ax.axhline(3.5, color=MUTED, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(285, 3.65, "▲ Native Engines: 2.0× – 4.1× WIN", color=ACCENT_EMERALD, fontsize=10.5, fontweight="bold", ha="right")
    ax.text(285, 3.15, "▼ llama.cpp GGUF: 0.45× – 0.58× LOSS", color=ACCENT_CORAL, fontsize=10.5, fontweight="bold", ha="right")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5, fontweight="bold", color=TEXT)
    ax.set_xlabel("Decode Throughput at ctx 512 (tok/s)", fontsize=12, labelpad=10)
    ax.set_xlim(0, 315)
    ax.grid(True, axis="x", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#475569", label="DSpark OFF (Baseline)"),
        mpatches.Patch(color=ACCENT_EMERALD, label="DSpark ON — Speedup (Native Engines)"),
        mpatches.Patch(color=ACCENT_CORAL, label="DSpark ON — Slowdown (llama.cpp 6–13% acceptance)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor=PANEL, edgecolor=GRID, fontsize=10.5)

    add_header_footer(
        fig,
        "The DSpark Speculative Decoding Reversal: llama.cpp vs Native Engines",
        "Same Inkling-2-Mini model • DSpark hurts llama.cpp everywhere, but unlocks 3.1×–4.1× on vLLM & SGLang",
        "*Strata Native Bench (BEAM-W4 TREE-6, 256/256 bit-exact parity); Strata v0.2.19 HTTP serve API does not engage DSpark (155.5 vs 155.5)."
    )
    save(fig, "02_dspark_reversal.png")


def chart_03_concurrency():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.76, top=0.84, bottom=0.12)

    levels = ["1", "4", "16", "64"]
    x = np.arange(len(levels))
    lookup = {(r["machine_id"], r["engine"]): [r["p1"], r["p4"], r["p16"], r["p64"]] for r in DATA["concurrency"]}

    series = [
        ("RTX PRO 6000 • Strata SQ-20", "rtx-6000", "Strata SQ-20", ACCENT_CYAN, "o", 3.2),
        ("RTX PRO 6000 • llama.cpp (IQ4_NL)", "rtx-6000", "llama.cpp IQ4_NL", ACCENT_BLUE, "^", 2.6),
        ("RTX Spark • SGLang (NVFP4)", "rtx-spark", "SGLang", ACCENT_EMERALD, "s", 3.0),
        ("RTX PRO 6000 • llama.cpp (Q4_K_XL)", "rtx-6000", "llama.cpp (600W)", "#93C5FD", "o", 2.2),
        ("DGX Spark • SGLang (NVFP4)", "dgx-spark", "SGLang", ACCENT_AMBER, "D", 2.6),
        ("RTX Spark • vLLM (NVFP4)", "rtx-spark", "vLLM", "#6EE7B7", "^", 2.2),
        ("DGX Spark • vLLM (NVFP4)", "dgx-spark", "vLLM", "#FDE68A", "s", 2.2),
        ("RTX 5090 • llama.cpp (Q4_K_XL)", "rtx-5090", "llama.cpp", ACCENT_ORANGE, "o", 2.4),
        ("DGX Spark • llama.cpp (Q4_K_XL)", "dgx-spark", "llama.cpp", MUTED, "--", 1.8),
        ("RTX PRO 6000 • Strata NVFP4 (#6520)", "rtx-6000", "Strata NVFP4 (#6520)", ACCENT_CORAL, "x", 2.4),
    ]

    for label, mid, eng, color, marker, lw in series:
        ys = lookup[(mid, eng)]
        if marker == "--":
            ax.plot(x, ys, linestyle="--", marker="o", color=color, linewidth=lw, markersize=6, label=label)
        else:
            ax.plot(x, ys, marker=marker, color=color, linewidth=lw, markersize=7.5, label=label)
        ax.annotate(f"{ys[-1]:.0f}", (x[-1], ys[-1]), textcoords="offset points", xytext=(8, -3),
                    fontsize=9.5, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}×" for l in levels], fontsize=12, fontweight="bold")
    ax.set_xlabel("Concurrent Parallel Requests (On-Box Load Generator, Zero Failed Workers)", fontsize=12, labelpad=10)
    ax.set_ylabel("Aggregate Output Throughput (tok/s)", fontsize=12, labelpad=10)
    ax.set_ylim(0, 1150)
    ax.grid(True, axis="y", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True,
              facecolor=PANEL, edgecolor=GRID, fontsize=10, labelspacing=0.8)

    add_header_footer(
        fig,
        "Concurrency Scaling (1 → 64 Parallel Streams): Strata & SGLang Break 900–1,047 tok/s",
        "All runs completed with 0 failed workers • Strata #6520 NVFP4 serializes requests (flat ~140 tok/s)",
        "RTX Spark SGLang (923 tok/s) matches single-card RTX PRO 6000 llama.cpp Q4_K_XL (907 tok/s) at 64 concurrent streams."
    )
    save(fig, "03_concurrency_scaling.png")


def chart_04_power_knee():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.82, bottom=0.14, wspace=0.22)

    caps = ["350 W\n(Throttled Cap)", "450 W\n(Sweet Spot)", "600 W\n(Card Max)"]
    x = np.arange(len(caps))
    w = 0.24

    p = DATA["power_ladder_rtx6000"]
    d512 = [r["a_decode_512"] for r in p]
    d131 = [r["a_decode_131k"] for r in p]
    soak = [r["h_decode"] for r in p]

    b1 = ax1.bar(x - w, d512, width=w, color=ACCENT_CYAN, label="Decode @512")
    b2 = ax1.bar(x, d131, width=w, color=ACCENT_VIOLET, label="Decode @128K")
    b3 = ax1.bar(x + w, soak, width=w, color=ACCENT_EMERALD, label="30m Soak Decode")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=TEXT)

    ax1.set_title("Speed Jumps +29% to +167% from 350W → 450W", fontsize=13.5, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(caps, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Decode Throughput (tok/s)", fontsize=11.5)
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", frameon=True, facecolor=PANEL, edgecolor=GRID, fontsize=10)

    # Right panel: Power & Temp Paradox
    mean_w = [r["h_mean_w"] for r in p]
    temp_c = [r["h_max_temp_c"] for r in p]
    eff = [r["h_tok_per_w"] for r in p]

    b_w = ax2.bar(x - 0.18, mean_w, width=0.34, color=ACCENT_AMBER, label="Mean Soak Power (W)")
    for bar in b_w:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 6, f"{h:.0f} W",
                 ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=ACCENT_AMBER)

    ax2_t = ax2.twinx()
    ax2_t.plot(x + 0.18, temp_c, marker="o", markersize=10, linewidth=3, color=ACCENT_CORAL, label="Max Soak Temp (°C)")
    for i, t in enumerate(temp_c):
        ax2_t.annotate(f"{t}°C\n({eff[i]:.3f} tok/W)", (x[i] + 0.18, t),
                       textcoords="offset points", xytext=(0, 12),
                       ha="center", fontsize=10, fontweight="bold", color=ACCENT_CORAL)

    ax2.set_title("The Power-Cap Paradox: Higher Cap = Cooler & Lower Mean Watts", fontsize=13.5, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(caps, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Power Draw During 30m Soak (W)", fontsize=11.5, color=ACCENT_AMBER)
    ax2_t.set_ylabel("Max GPU Temperature (°C)", fontsize=11.5, color=ACCENT_CORAL)
    ax2.set_ylim(0, 410)
    ax2_t.set_ylim(45, 102)
    ax2.grid(True, axis="y", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2_t.spines["top"].set_visible(False)

    add_header_footer(
        fig,
        "RTX PRO 6000 Blackwell Power Ladder (350 W vs 450 W vs 600 W)",
        "Same card, same weights, same harness • Escaping the 350 W cap removes SW Power throttling and boosts efficiency 2.5×",
        "At 350 W, SW Power Cap pins clocks at ~1800 MHz (high duty cycle, 15% soak fade). At 450/600 W, workload peaks at 230 W and finishes faster."
    )
    save(fig, "04_rtx6000_power_knee.png")


def chart_05_rtx6000_engines():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.82, bottom=0.16, wspace=0.22)

    arms = [
        "Strata\nSQ-20",
        "Strata NVFP4\n(#6520)",
        "llama.cpp\nIQ4_NL",
        "Strata GGUF\n(#6642)",
        "llama.cpp\nQ4_K_XL",
    ]
    x = np.arange(len(arms))
    d512 = [178.6, 163.1, 95.4, 93.6, 86.8]
    d128k = [95.0, 96.0, 63.6, 57.1, 64.0]
    p64 = [1046.6, 140.3, 978.0, 127.0, 906.8]
    evals = [0.800, 0.867, 0.733, 0.533, 0.800]

    w = 0.34
    b1 = ax1.bar(x - w / 2, d512, width=w, color=ACCENT_CYAN, label="Decode @512")
    b2 = ax1.bar(x + w / 2, d128k, width=w, color=ACCENT_VIOLET, label="Decode @128K")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h:.0f}",
                     ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax1.set_title("Single-Stream Decode (tok/s)", fontsize=13.5, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Decode Throughput (tok/s)", fontsize=11.5)
    ax1.set_ylim(0, 210)
    ax1.grid(True, axis="y", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", frameon=True, facecolor=PANEL, edgecolor=GRID, fontsize=10)

    # Right panel: Concurrency @64 + Eval badge
    colors_p64 = [ACCENT_EMERALD if v > 800 else ACCENT_CORAL for v in p64]
    b3 = ax2.bar(x, p64, width=0.52, color=colors_p64)
    for i, bar in enumerate(b3):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 22,
                 f"{h:.0f} tok/s\nEval: {evals[i]:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT)

    ax2.set_title("Batch Serving @64 Parallel & Combined Eval Score", fontsize=13.5, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(arms, fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Aggregate Output @64 Workers (tok/s)", fontsize=11.5)
    ax2.set_ylim(0, 1250)
    ax2.grid(True, axis="y", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    add_header_footer(
        fig,
        "RTX PRO 6000 (600 W): Engine & Quant Trade-Offs (llama.cpp vs Strata Arms)",
        "Strata SQ-20 wins speed & batching (but OOMs at 1M) • #6520 NVFP4 wins quality (0.867) but doesn't batch • #6642 GGUF trails llama.cpp",
        "Combined Eval = 20 Math + 10 Sandboxed Code items (thinking off). Code = 10/10 across all five arms."
    )
    save(fig, "05_rtx6000_engines_and_quants.png")


def chart_06_1m_breakthrough():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.25, right=0.92, top=0.82, bottom=0.14)

    rows = [
        ("DGX Spark • vLLM max-bf16\n+ Assistant Prefill (Bypass Loop)", 100, ACCENT_EMERALD, "3/3 (100%)  •  9 output tokens  •  clean stop!"),
        ("DGX Spark • vLLM max-fp8\nStandard Chat (reasoning_effort=none)", 0, ACCENT_CORAL, "0/3 content  •  budget exhausted (1/3 in reasoning)"),
        ("DGX Spark • SGLang max\nStandard Chat (reasoning_effort=none)", 0, ACCENT_CORAL, "0/3 content  •  budget exhausted (2/3 in reasoning)"),
        ("RTX Spark • vLLM & SGLang\nStandard Chat (reasoning_effort=none)", 0, ACCENT_CORAL, "0/3 content  •  reproduces DGX Spark run-for-run"),
        ("DGX / RTX Spark / RTX 6000 • llama.cpp\nStandard Chat (2k & 8k probes)", 0, ACCENT_CORAL, "0/3 content  •  degenerate repetition loop at 1M"),
        ("RTX PRO 6000 • Strata SQ-20\nStandard Chat (1,010,991 tokens)", 0, "#64748B", "BLOCKED  •  CUDA OOM at 267K prefill"),
    ]
    rows = rows[::-1]
    y = np.arange(len(rows))

    vals = [r[1] if r[1] > 0 else 2.2 for r in rows]  # small stub so 0% bars are visible
    colors = [r[2] for r in rows]
    labels = [r[0] for r in rows]
    notes = [r[3] for r in rows]

    ax.barh(y, vals, height=0.56, color=colors)
    for i, (val, note, color) in enumerate(zip(vals, notes, colors)):
        x_pos = val + 2.5 if val < 50 else 50
        ha = "left" if val < 50 else "center"
        txt_color = BG if val >= 50 else color
        ax.text(x_pos, y[i], note, va="center", ha=ha, fontsize=11.5, fontweight="bold", color=txt_color)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold", color=TEXT)
    ax.set_xlabel("1M-Context (1,047,040 Tokens) Exact Needle Retrieval Accuracy (%)", fontsize=12, labelpad=10)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.grid(True, axis="x", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_header_footer(
        fig,
        "The 1M-Context Breakthrough: Inkling-2-Mini CAN Retrieve at 1.047M Tokens",
        "Standard chat fails 0/3 everywhere because the template ignores reasoning_effort=none (~98.4% reasoning tokens) and loops",
        "Fix: chat-assistant-prefill + continue_final_message bypasses the reasoning loop → 3/3 exact hits (XJ9-4782, XJ9-6498, XJ9-7378) in ~25 min/run."
    )
    save(fig, "06_1m_context_breakthrough.png")


def main():
    chart_01_single_stream()
    chart_02_dspark_reversal()
    chart_03_concurrency()
    chart_04_power_knee()
    chart_05_rtx6000_engines()
    chart_06_1m_breakthrough()


if __name__ == "__main__":
    main()
