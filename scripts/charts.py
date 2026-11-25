"""Chart primitives drawn as deterministic SVG, matching GumbiiDigital/omlx-fleet-bench.

Design rules:
- Thin horizontal bars from one baseline with the value at the tip
- One scale per panel (or shared scale where specified)
- Text in ink / secondary / muted colours (never series colour)
- Hover tooltip on every mark
- 840px width that stays crisp at GitHub README column width
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from svg import Canvas, Theme, text_width

WIDTH = 840
PAD = 24
PANEL_GAP = 32
ROW_PITCH = 26
BAR_HEIGHT = 14
PANEL_TITLE_HEIGHT = 26
AXIS_LABEL_HEIGHT = 20
PANEL_ROW_GAP = 22
LINE_PLOT_HEIGHT = 300
LEGEND_ROW_HEIGHT = 18
PANEL_PLOT_HEIGHT = 210

TITLE_SIZE = 16
SUBTITLE_SIZE = 12
CAPTION_WIDTH = (WIDTH - 2 * PAD) * 0.92
PANEL_TITLE_SIZE = 13
LABEL_SIZE = 12
VALUE_SIZE = 12
NOTE_SIZE = 11
FOOTNOTE_SIZE = 11
LINE_HEIGHT = 17


@dataclass(frozen=True)
class BarRow:
    label: str
    value: float
    value_text: str
    series: str
    note: str = ""


@dataclass(frozen=True)
class BarPanel:
    title: str
    rows: Sequence[BarRow]


@dataclass(frozen=True)
class LinePoint:
    x: float
    y: float
    x_text: str
    y_text: str


@dataclass(frozen=True)
class LineSeries:
    label: str
    series: str
    points: Sequence[LinePoint]


@dataclass(frozen=True)
class LinePanel:
    title: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_ticks: Sequence[Tuple[float, str]]
    y_ticks: Sequence[Tuple[float, str]]
    series: Sequence[LineSeries]
    x_label: str = ""
    y_label: str = ""
    log_x: bool = True
    log_y: bool = False


@dataclass(frozen=True)
class Caption:
    title: str
    subtitle: Sequence[str] = ()
    footnotes: Sequence[str] = ()

    @property
    def label(self) -> str:
        return " ".join([self.title, *self.subtitle])


def _wrap(text: str, size: float, width: float) -> List[str]:
    lines: List[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and text_width(candidate, size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines or [""]


def _subtitle_lines(caption: Caption) -> List[str]:
    return [line for entry in caption.subtitle for line in _wrap(entry, SUBTITLE_SIZE, CAPTION_WIDTH)]


def _footnote_lines(caption: Caption) -> List[str]:
    return [line for entry in caption.footnotes for line in _wrap(entry, FOOTNOTE_SIZE, CAPTION_WIDTH)]


def _header_height(caption: Caption) -> float:
    return PAD + TITLE_SIZE + 6 + len(_subtitle_lines(caption)) * LINE_HEIGHT + 12


def _footer_height(caption: Caption) -> float:
    if not caption.footnotes:
        return PAD - 8
    return 6 + len(_footnote_lines(caption)) * (FOOTNOTE_SIZE + 5) + PAD - 8


def _columns_that_fit(count: int, columns: int, needed: float) -> int:
    columns = max(1, min(columns, count))
    while columns > 1:
        width = (WIDTH - 2 * PAD - (columns - 1) * PANEL_GAP) / columns
        if width >= needed:
            break
        columns -= 1
    return columns


def _grid(count: int, columns: int, panel_height: float, top: float) -> Tuple[List[Tuple[float, float]], float, float]:
    columns = max(1, min(columns, count))
    panel_width = (WIDTH - 2 * PAD - (columns - 1) * PANEL_GAP) / columns
    origins = []
    for index in range(count):
        col, row = index % columns, index // columns
        origins.append((PAD + col * (panel_width + PANEL_GAP), top + row * (panel_height + PANEL_ROW_GAP)))
    grid_rows = (count + columns - 1) // columns
    bottom = top + grid_rows * panel_height + (grid_rows - 1) * PANEL_ROW_GAP
    return origins, panel_width, bottom


def _open(
    caption: Caption,
    theme: Theme,
    height: float,
    *,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> Canvas:
    canvas = Canvas(WIDTH, height, theme, label=caption.label, x_scale=x_scale, y_scale=y_scale)
    canvas.rect(0.5, 0.5, WIDTH - 1, height - 1, theme.surface, radius=8, stroke=theme.border)
    y = PAD + TITLE_SIZE
    canvas.text(PAD, y, caption.title, size=TITLE_SIZE, weight=600)
    for line in _subtitle_lines(caption):
        y += LINE_HEIGHT
        canvas.text(PAD, y, line, size=SUBTITLE_SIZE, fill=theme.secondary)
    return canvas


def _close(canvas: Canvas, caption: Caption, grid_bottom: float) -> str:
    y = grid_bottom + 6
    for line in _footnote_lines(caption):
        y += FOOTNOTE_SIZE + 5
        canvas.text(PAD, y, line, size=FOOTNOTE_SIZE, fill=canvas.theme.muted)
    return canvas.render()


def _label_width(labels: Sequence[str]) -> float:
    return max([text_width(label, LABEL_SIZE) for label in labels] + [60.0]) + 14


def _value_width(rows: Sequence[BarRow]) -> float:
    widest = 0.0
    for row in rows:
        width = text_width(row.value_text, VALUE_SIZE, bold=True)
        if row.note:
            width += 8 + text_width(row.note, NOTE_SIZE)
        widest = max(widest, width)
    return widest + 14


def _value_label(canvas: Canvas, x: float, baseline: float, row: BarRow) -> None:
    canvas.text(x, baseline, row.value_text, size=VALUE_SIZE, weight=600)
    if row.note:
        canvas.text(
            x + text_width(row.value_text, VALUE_SIZE, bold=True) + 8,
            baseline,
            row.note,
            size=NOTE_SIZE,
            fill=canvas.theme.muted,
        )


def bar_panels(
    caption: Caption,
    panels: Sequence[BarPanel],
    theme: Theme,
    *,
    columns: int = 2,
    axis_label: str = "",
    shared_scale: bool = False,
) -> str:
    if not panels:
        raise ValueError("no panels")
    all_rows = [row for panel in panels for row in panel.rows]
    label_w = _label_width([row.label for row in all_rows])
    value_w = _value_width(all_rows)
    panel_height = PANEL_TITLE_HEIGHT + max(len(panel.rows) for panel in panels) * ROW_PITCH
    top = _header_height(caption)
    columns = _columns_that_fit(len(panels), columns, label_w + value_w + 48)
    origins, panel_width, bottom = _grid(len(panels), columns, panel_height, top)
    track = panel_width - label_w - value_w
    if track < 40:
        raise ValueError("labels leave no room for the bars; use fewer columns")
    room = AXIS_LABEL_HEIGHT if axis_label else 0.0
    global_scale = max((row.value for row in all_rows), default=1.0) or 1.0

    canvas = _open(caption, theme, bottom + room + _footer_height(caption))
    for panel, (px, py) in zip(panels, origins):
        canvas.text(px, py + PANEL_TITLE_SIZE, panel.title, size=PANEL_TITLE_SIZE, weight=600)
        rows_top = py + PANEL_TITLE_HEIGHT
        x0 = px + label_w
        scale = global_scale if shared_scale else (max((row.value for row in panel.rows), default=0.0) or 1.0)
        for index, row in enumerate(panel.rows):
            y = rows_top + index * ROW_PITCH
            baseline = y + BAR_HEIGHT - 2.5
            canvas.text(px, baseline, row.label, size=LABEL_SIZE, fill=theme.secondary)
            width = max(row.value, 0.0) / scale * track
            canvas.bar(
                x0,
                y,
                width,
                BAR_HEIGHT,
                theme.series[row.series],
                tooltip=f"{panel.title} · {row.label}: {row.value_text} {row.note}".strip(),
            )
            _value_label(canvas, x0 + width + 8, baseline, row)
        canvas.line(x0, rows_top - 4, x0, rows_top + len(panel.rows) * ROW_PITCH - 8, theme.axis)
    if axis_label:
        for left in sorted({px for px, _ in origins}):
            canvas.text(left + panel_width / 2, bottom + 16, axis_label, size=NOTE_SIZE, fill=theme.muted, anchor="middle")
    return _close(canvas, caption, bottom + room)


def _legend(canvas: Canvas, series: Sequence[LineSeries], y: float) -> int:
    """Series names in wrapped rows above the plot, returns number of rows used."""
    x = PAD
    rows = 1
    max_x = WIDTH - PAD
    for line in series:
        item_w = 20 + text_width(line.label, NOTE_SIZE) + 22
        if x > PAD and x + item_w > max_x:
            x = PAD
            y += LEGEND_ROW_HEIGHT
            rows += 1
        canvas.line(x, y - 4, x + 14, y - 4, canvas.theme.series[line.series], width=2.2, round_cap=True)
        canvas.text(x + 20, y, line.label, size=NOTE_SIZE, fill=canvas.theme.secondary)
        x += item_w
    return rows


def _span(low: float, high: float, is_log: bool):
    if is_log:
        floor, span = math.log10(low), math.log10(high) - math.log10(low)
        return lambda value: (math.log10(min(max(value, low), high)) - floor) / span
    span = high - low
    return lambda value: (min(max(value, low), high) - low) / span


def _place_value(
    canvas: Canvas,
    text: str,
    x: float,
    y: float,
    plot_left: float,
    plot_right: float,
    plot_bottom: float,
    placed: List[Tuple[float, float, float, float]],
) -> None:
    wide = text_width(text, NOTE_SIZE)
    boxes = tuple(
        (x + gap, y + lift, anchor)
        for lift in (-8, -21, -34, 15, 28, -47, 41)
        for gap, anchor in ((8, "start"), (-8, "end"))
    )
    for cx, cy, anchor in boxes:
        left = cx if anchor == "start" else cx - wide
        right = left + wide
        top, bottom = cy - NOTE_SIZE, cy + 2
        if left < plot_left or right > plot_right or bottom > plot_bottom - 2:
            continue
        if any(not (right < hl or left > hr or bottom < ht or top > hb) for hl, ht, hr, hb in placed):
            continue
        placed.append((left, top, right, bottom))
        canvas.text(cx, cy, text, size=NOTE_SIZE, fill=canvas.theme.muted, anchor=anchor)
        return
    cx = min(max(x + 8, plot_left), plot_right - 4 - wide)
    placed.append((cx, y - 8 - NOTE_SIZE, cx + wide, y - 6))
    canvas.text(cx, y - 8, text, size=NOTE_SIZE, fill=canvas.theme.muted)


def line_panel(caption: Caption, panel: LinePanel, theme: Theme) -> str:
    gutter = max([text_width(label, NOTE_SIZE) for _, label in panel.y_ticks] + [28.0]) + 12
    axis_room = 18.0 if panel.y_label else 0.0
    left, right = PAD + gutter + axis_room, WIDTH - PAD - 24

    # Estimate legend rows
    x_cur = PAD
    legend_rows = 1
    for line in panel.series:
        w = 20 + text_width(line.label, NOTE_SIZE) + 22
        if x_cur > PAD and x_cur + w > WIDTH - PAD:
            x_cur = PAD
            legend_rows += 1
        x_cur += w

    top = _header_height(caption) + legend_rows * LEGEND_ROW_HEIGHT + 8
    bottom = top + LINE_PLOT_HEIGHT
    room = AXIS_LABEL_HEIGHT if panel.x_label else 0

    canvas = _open(
        caption,
        theme,
        bottom + LEGEND_ROW_HEIGHT + room + _footer_height(caption),
        x_scale="log" if panel.log_x else "linear",
        y_scale="log" if panel.log_y else "linear",
    )
    _legend(canvas, panel.series, _header_height(caption) + 6)

    across = _span(panel.x_min, panel.x_max, panel.log_x)
    down = _span(panel.y_min, panel.y_max, panel.log_y)

    def x_of(val: float) -> float:
        return left + across(val) * (right - left)

    def y_of(val: float) -> float:
        return bottom - down(val) * (bottom - top)

    for value, label in panel.y_ticks:
        y = y_of(value)
        canvas.line(left, y, right, y, theme.grid)
        canvas.text(left - 8, y + 4, label, size=NOTE_SIZE, fill=theme.muted, anchor="end")
    for value, label in panel.x_ticks:
        x = x_of(value)
        canvas.line(x, top, x, bottom, theme.grid)
        canvas.text(x, bottom + 14, label, size=NOTE_SIZE, fill=theme.muted, anchor="middle")
    canvas.line(left, top, left, bottom, theme.axis)
    canvas.line(left, bottom, right, bottom, theme.axis)

    placed: List[Tuple[float, float, float, float]] = []
    for series in panel.series:
        colour = theme.series[series.series]
        pts = [(x_of(pt.x), y_of(pt.y)) for pt in series.points]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            canvas.line(x1, y1, x2, y2, colour, width=2, round_cap=True)
        for pt, (x, y) in zip(series.points, pts):
            canvas.dot(
                x, y, 4, colour, ring=theme.surface,
                tooltip=f"{series.label} · {pt.x_text}: {pt.y_text}",
            )
    for series in panel.series:
        for pt in series.points:
            _place_value(canvas, pt.y_text, x_of(pt.x), y_of(pt.y), left, right, bottom, placed)

    if panel.y_label:
        canvas.text(PAD + 4, (top + bottom) / 2, panel.y_label, size=NOTE_SIZE, fill=theme.muted, anchor="middle", rotate=-90)
    if panel.x_label:
        canvas.text((left + right) / 2, bottom + 30, panel.x_label, size=NOTE_SIZE, fill=theme.muted, anchor="middle")

    return _close(canvas, caption, bottom + LEGEND_ROW_HEIGHT + room - 6)
