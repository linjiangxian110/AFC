"""Draw Figure 1 concept schematic for the TrustMoE-Traj paper.

The figure is intentionally schematic rather than architectural: it contrasts
best-of-K evaluation with set-level quality-diversity refinement.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle


WIDTH_MM = 183
HEIGHT_MM = 112
OUT_DIR = Path("figures") / "figure1_core_idea"
OUT_STEM = OUT_DIR / "figure1_core_idea"


COLORS = {
    "ink": "#1f252d",
    "muted": "#66727f",
    "grid": "#d8dee6",
    "panel": "#fbfcfd",
    "gray": "#aeb6c1",
    "gray_dark": "#7b8491",
    "gt": "#2f9e44",
    "best": "#199fb1",
    "bad": "#d95f43",
    "warn": "#e6a23c",
    "analog": "#3a86c8",
    "refine": "#7567c9",
    "refine2": "#2a9d8f",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def bezier(start, c1, c2, end, n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    start = np.asarray(start, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    end = np.asarray(end, dtype=float)
    return (
        (1.0 - t) ** 3 * start
        + 3.0 * (1.0 - t) ** 2 * t * c1
        + 3.0 * (1.0 - t) * t**2 * c2
        + t**3 * end
    )


def add_curve(
    ax,
    start,
    end,
    *,
    color,
    lw=1.0,
    alpha=1.0,
    ls="-",
    c1=None,
    c2=None,
    zorder=3,
    end_marker=True,
):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if c1 is None:
        c1 = start + np.array([0.18, 0.02])
    if c2 is None:
        c2 = end - np.array([0.18, 0.02])
    pts = bezier(start, c1, c2, end)
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=color,
        lw=lw,
        alpha=alpha,
        ls=ls,
        solid_capstyle="round",
        zorder=zorder,
    )
    if end_marker:
        ax.plot(
            end[0],
            end[1],
            marker="o",
            ms=max(1.8, lw * 1.45),
            color=color,
            alpha=alpha,
            markeredgewidth=0,
            zorder=zorder + 1,
        )
    return pts


def add_pedestrian(ax, xy, scale=1.0, color=None, zorder=10):
    color = color or COLORS["ink"]
    x, y = xy
    ax.add_patch(Circle((x, y + 0.040 * scale), 0.018 * scale, fc=color, ec="none", zorder=zorder))
    ax.plot([x, x], [y + 0.018 * scale, y - 0.045 * scale], color=color, lw=1.2, zorder=zorder)
    ax.plot([x, x - 0.030 * scale], [y - 0.005 * scale, y - 0.030 * scale], color=color, lw=1.0, zorder=zorder)
    ax.plot([x, x + 0.030 * scale], [y - 0.005 * scale, y - 0.030 * scale], color=color, lw=1.0, zorder=zorder)
    ax.plot([x, x - 0.028 * scale], [y - 0.045 * scale, y - 0.088 * scale], color=color, lw=1.0, zorder=zorder)
    ax.plot([x, x + 0.030 * scale], [y - 0.045 * scale, y - 0.088 * scale], color=color, lw=1.0, zorder=zorder)


def add_panel_frame(ax, label: str, title: str, subtitle: str):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.010,rounding_size=0.020",
            fc=COLORS["panel"],
            ec=COLORS["grid"],
            lw=0.8,
            clip_on=False,
            zorder=0,
        )
    )
    ax.text(0.035, 0.945, label, fontsize=8.2, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.105, 0.945, title, fontsize=8.0, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(
        0.105,
        0.895,
        "\n".join(textwrap.wrap(subtitle, width=38)),
        fontsize=5.8,
        color=COLORS["muted"],
        va="top",
        linespacing=1.12,
    )


def add_chip(ax, xy, text, fc, ec=None, color=None, width=None, fontsize=5.9):
    x, y = xy
    color = color or COLORS["ink"]
    if width is None:
        width = 0.014 * len(text) + 0.050
    height = 0.058
    ax.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.020",
            fc=fc,
            ec=ec or fc,
            lw=0.5,
            zorder=12,
        )
    )
    ax.text(x + width / 2, y, text, fontsize=fontsize, color=color, ha="center", va="center", zorder=13)


def add_section_label(ax, xy, text, color=None):
    ax.text(
        xy[0],
        xy[1],
        text,
        fontsize=6.4,
        fontweight="bold",
        color=color or COLORS["ink"],
        ha="left",
        va="center",
    )


def draw_observed(ax, start=(0.12, 0.48), end=(0.34, 0.54), scale=1.0):
    pts = np.array(
        [
            [start[0], start[1]],
            [start[0] + 0.055 * scale, start[1] + 0.018 * scale],
            [start[0] + 0.120 * scale, start[1] + 0.030 * scale],
            [end[0] - 0.045 * scale, end[1] + 0.010 * scale],
            [end[0], end[1]],
        ]
    )
    ax.plot(pts[:, 0], pts[:, 1], color=COLORS["ink"], lw=1.8, solid_capstyle="round", zorder=8)
    ax.scatter(pts[::2, 0], pts[::2, 1], s=5.5, color=COLORS["ink"], zorder=9)
    add_pedestrian(ax, (end[0] - 0.01, end[1] + 0.075 * scale), scale=0.58 * scale)
    return np.asarray(end)


def draw_left_panel(ax):
    add_panel_frame(
        ax,
        "a",
        "Best-of-K view",
        "A single matching candidate can dominate the reported score.",
    )
    origin = draw_observed(ax, start=(0.10, 0.43), end=(0.34, 0.50), scale=1.0)

    candidates = [
        ((0.84, 0.78), (0.48, 0.56), (0.66, 0.72)),
        ((0.87, 0.65), (0.52, 0.57), (0.68, 0.60)),
        ((0.86, 0.53), (0.50, 0.52), (0.69, 0.50)),
        ((0.83, 0.37), (0.49, 0.45), (0.66, 0.34)),
        ((0.80, 0.25), (0.50, 0.42), (0.66, 0.24)),
        ((0.88, 0.88), (0.52, 0.60), (0.72, 0.86)),
        ((0.76, 0.16), (0.50, 0.39), (0.62, 0.15)),
        ((0.92, 0.45), (0.54, 0.50), (0.72, 0.42)),
    ]
    for end, c1, c2 in candidates:
        add_curve(ax, origin, end, c1=c1, c2=c2, color=COLORS["gray"], lw=0.9, alpha=0.78, zorder=2)

    gt_end = (0.82, 0.66)
    add_curve(
        ax,
        origin,
        gt_end,
        c1=(0.49, 0.56),
        c2=(0.66, 0.64),
        color=COLORS["gt"],
        lw=1.4,
        alpha=0.88,
        ls=(0, (3, 2)),
        zorder=4,
        end_marker=False,
    )
    ax.plot(gt_end[0], gt_end[1], marker="*", ms=7.5, color=COLORS["gt"], markeredgecolor="white", markeredgewidth=0.5, zorder=7)
    add_curve(
        ax,
        origin,
        (0.80, 0.64),
        c1=(0.48, 0.54),
        c2=(0.64, 0.63),
        color=COLORS["best"],
        lw=1.9,
        alpha=1.0,
        zorder=6,
    )

    add_chip(ax, (0.55, 0.165), "only best trajectory is rewarded", "#e9f6f8", COLORS["best"], COLORS["best"], width=0.38)
    ax.text(0.77, 0.71, "GT", fontsize=5.7, color=COLORS["gt"], ha="right", va="bottom")
    ax.text(0.12, 0.315, "observed past", fontsize=5.7, color=COLORS["ink"], ha="left")
    ax.text(0.735, 0.775, "other K-1 candidates\nare weakly audited", fontsize=5.6, color=COLORS["muted"], ha="center")


def draw_middle_panel(ax):
    add_panel_frame(
        ax,
        "b",
        "Set-level failures",
        "Best-of-K metrics can hide poor candidates or collapsed modes.",
    )

    # Top: best-of-K success, set-level failure.
    add_section_label(ax, (0.075, 0.760), "Best hit, poor set")
    origin = draw_observed(ax, start=(0.09, 0.61), end=(0.31, 0.66), scale=0.72)
    gt_end = (0.78, 0.76)
    add_curve(ax, origin, gt_end, c1=(0.45, 0.70), c2=(0.62, 0.76), color=COLORS["gt"], lw=1.25, ls=(0, (3, 2)), end_marker=False)
    ax.plot(gt_end[0], gt_end[1], marker="*", ms=6.8, color=COLORS["gt"], markeredgecolor="white", markeredgewidth=0.45, zorder=7)
    add_curve(ax, origin, (0.77, 0.735), c1=(0.45, 0.68), c2=(0.62, 0.73), color=COLORS["best"], lw=1.6, zorder=6)
    bad_curves = [
        ((0.86, 0.90), (0.44, 0.84), (0.68, 0.96)),
        ((0.88, 0.57), (0.44, 0.62), (0.68, 0.50)),
        ((0.70, 0.52), (0.44, 0.59), (0.61, 0.43)),
        ((0.92, 0.68), (0.46, 0.75), (0.73, 0.64)),
    ]
    for i, (end, c1, c2) in enumerate(bad_curves):
        add_curve(ax, origin, end, c1=c1, c2=c2, color=COLORS["bad"] if i < 2 else COLORS["warn"], lw=1.0, alpha=0.78, zorder=3)
    ax.text(0.61, 0.545, "implausible\nor redundant", fontsize=5.4, color=COLORS["bad"], ha="center", va="center")
    ax.text(0.84, 0.80, "min score OK", fontsize=5.3, color=COLORS["best"], ha="left")

    ax.plot([0.06, 0.94], [0.485, 0.485], color=COLORS["grid"], lw=0.8)

    # Bottom: single-GT over-optimization.
    add_section_label(ax, (0.075, 0.420), "Single-GT compression")
    origin2 = draw_observed(ax, start=(0.09, 0.235), end=(0.31, 0.275), scale=0.72)
    true_modes = [
        ((0.78, 0.405), (0.44, 0.34), (0.62, 0.40)),
        ((0.80, 0.270), (0.45, 0.28), (0.63, 0.26)),
        ((0.74, 0.145), (0.44, 0.22), (0.61, 0.14)),
    ]
    for end, c1, c2 in true_modes:
        add_curve(ax, origin2, end, c1=c1, c2=c2, color=COLORS["analog"], lw=0.95, alpha=0.32, ls=(0, (3, 2)), zorder=2)
    for j, off in enumerate([-0.030, -0.014, 0.000, 0.018, 0.036]):
        add_curve(
            ax,
            origin2,
            (0.795 + 0.008 * np.sin(j), 0.395 + off * 0.20),
            c1=(0.45, 0.32 + off),
            c2=(0.63, 0.38 + off * 0.30),
            color=COLORS["warn"],
            lw=1.05,
            alpha=0.82,
            zorder=5,
        )
    ax.text(0.62, 0.185, "mode coverage shrinks", fontsize=5.4, color=COLORS["warn"], ha="center")
    ax.text(0.79, 0.445, "single GT", fontsize=5.3, color=COLORS["gt"], ha="center")


def draw_right_panel(ax):
    add_panel_frame(
        ax,
        "c",
        "Quality-diversity refinement",
        "Refine the whole prediction set toward plausible future modes.",
    )
    origin = draw_observed(ax, start=(0.08, 0.43), end=(0.31, 0.50), scale=0.98)

    analog_modes = [
        ((0.83, 0.76), (0.47, 0.60), (0.65, 0.77)),
        ((0.87, 0.55), (0.48, 0.52), (0.68, 0.54)),
        ((0.80, 0.30), (0.47, 0.42), (0.65, 0.28)),
    ]
    for end, c1, c2 in analog_modes:
        add_curve(
            ax,
            origin,
            end,
            c1=c1,
            c2=c2,
            color=COLORS["analog"],
            lw=1.15,
            alpha=0.86,
            ls=(0, (3, 2)),
            zorder=3,
        )
    ax.text(0.72, 0.845, "analogical futures", fontsize=5.6, color=COLORS["analog"], ha="center")

    original = [
        ((0.79, 0.86), (0.48, 0.62), (0.66, 0.88)),
        ((0.92, 0.67), (0.50, 0.57), (0.72, 0.68)),
        ((0.88, 0.47), (0.50, 0.50), (0.70, 0.45)),
        ((0.73, 0.20), (0.48, 0.38), (0.62, 0.18)),
        ((0.90, 0.26), (0.52, 0.38), (0.70, 0.25)),
    ]
    for end, c1, c2 in original:
        add_curve(ax, origin, end, c1=c1, c2=c2, color=COLORS["gray"], lw=0.88, alpha=0.46, zorder=2)

    refined = [
        ((0.82, 0.735), (0.48, 0.58), (0.65, 0.73), COLORS["refine"]),
        ((0.86, 0.555), (0.48, 0.52), (0.68, 0.55), COLORS["refine2"]),
        ((0.79, 0.315), (0.48, 0.43), (0.65, 0.30), COLORS["refine"]),
    ]
    for end, c1, c2, color in refined:
        add_curve(ax, origin, end, c1=c1, c2=c2, color=color, lw=1.85, alpha=1.0, zorder=7)

    arrow = FancyArrowPatch(
        (0.53, 0.260),
        (0.61, 0.335),
        arrowstyle="-|>",
        mutation_scale=8,
        lw=0.9,
        color=COLORS["refine"],
        alpha=0.85,
        zorder=10,
    )
    ax.add_patch(arrow)
    ax.text(0.490, 0.230, "distribution-level\nresidual refinement", fontsize=5.4, color=COLORS["refine"], ha="center")

    add_chip(ax, (0.125, 0.130), "avgADE/FDE", "#f0edf9", COLORS["refine"], COLORS["refine"], width=0.205, fontsize=5.6)
    add_chip(ax, (0.395, 0.130), "AFC recall", "#e8f3fb", COLORS["analog"], COLORS["analog"], width=0.190, fontsize=5.6)
    add_chip(ax, (0.665, 0.130), "mode coverage", "#e9f7f3", COLORS["refine2"], COLORS["refine2"], width=0.230, fontsize=5.6)
    ax.text(0.68, 0.060, "accurate, diverse, plausible", fontsize=6.0, color=COLORS["ink"], fontweight="bold", ha="center")


def draw_top_flow(fig, axes):
    overlay = fig.add_axes([0, 0, 1, 1], zorder=20)
    overlay.axis("off")
    y = 0.905
    for left_ax, right_ax in [(axes[0], axes[1]), (axes[1], axes[2])]:
        b1 = left_ax.get_position()
        b2 = right_ax.get_position()
        start = (b1.x1 + 0.010, y)
        end = (b2.x0 - 0.010, y)
        overlay.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=9,
                lw=0.8,
                color=COLORS["gray_dark"],
                alpha=0.85,
            )
        )


def add_footer(fig):
    fig.text(
        0.50,
        0.037,
        "Refine the whole multimodal prediction distribution, not only the best trajectory.",
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["ink"],
        fontweight="bold",
    )


def draw_figure():
    configure_matplotlib()
    width_in = WIDTH_MM / 25.4
    height_in = HEIGHT_MM / 25.4
    fig = plt.figure(figsize=(width_in, height_in))
    fig.text(
        0.5,
        0.965,
        "Beyond Best-of-K: Quality of the Whole Multimodal Prediction Set",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.5,
        0.925,
        "From single-trajectory success to set-level accuracy, plausible diversity, and coverage.",
        ha="center",
        va="top",
        fontsize=6.7,
        color=COLORS["muted"],
    )

    lefts = [0.045, 0.365, 0.685]
    axes = [fig.add_axes([left, 0.115, 0.270, 0.755]) for left in lefts]
    draw_left_panel(axes[0])
    draw_middle_panel(axes[1])
    draw_right_panel(axes[2])
    draw_top_flow(fig, axes)
    add_footer(fig)
    return fig


def save_outputs(fig) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT_STEM}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT_STEM}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


def main() -> None:
    fig = draw_figure()
    save_outputs(fig)
    plt.close(fig)
    print(f"saved: {OUT_STEM}.svg")
    print(f"saved: {OUT_STEM}.pdf")
    print(f"saved: {OUT_STEM}.png")
    print(f"saved: {OUT_STEM}.tiff")


if __name__ == "__main__":
    main()
