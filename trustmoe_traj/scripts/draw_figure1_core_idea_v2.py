"""Draw Figure 1 v2 concept schematic with scene-aware plausible diversity.

Version 2 keeps the original best-of-K argument but adds obstacles, road
boundaries, and other pedestrians to make implausible candidates visually clear.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trustmoe_traj.scripts.draw_figure1_core_idea import (
    COLORS,
    HEIGHT_MM,
    WIDTH_MM,
    add_chip,
    add_curve,
    add_panel_frame,
    add_pedestrian,
    configure_matplotlib,
    draw_observed,
)


OUT_DIR = Path("figures") / "figure1_core_idea_v2"
OUT_STEM = OUT_DIR / "figure1_core_idea_v2"


def add_scene(ax, *, compact: bool = False, show_other_label: bool = True) -> None:
    """Draw a minimal shared scene: walkable area, obstacle, and another agent."""
    # Walkable corridor.
    ax.add_patch(
        Rectangle(
            (0.045, 0.185),
            0.910,
            0.610,
            fc="#f7f9fb",
            ec="none",
            zorder=0.5,
        )
    )
    ax.plot([0.055, 0.940], [0.795, 0.795], color="#cfd6df", lw=0.9, zorder=1)
    ax.plot([0.055, 0.940], [0.185, 0.185], color="#cfd6df", lw=0.9, zorder=1)
    ax.plot([0.070, 0.925], [0.490, 0.490], color="#e5e9ef", lw=0.7, ls=(0, (5, 5)), zorder=1)

    # Static obstacle.
    obs_xy = (0.555, 0.435 if not compact else 0.415)
    obs_w, obs_h = 0.120, 0.145
    ax.add_patch(
        Rectangle(
            obs_xy,
            obs_w,
            obs_h,
            fc="#eef1f4",
            ec="#9aa5b1",
            lw=0.8,
            hatch="///",
            zorder=6,
        )
    )
    ax.text(
        obs_xy[0] + obs_w / 2,
        obs_xy[1] - 0.028,
        "obstacle",
        fontsize=5.0,
        color=COLORS["gray_dark"],
        ha="center",
        va="top",
        zorder=8,
    )

    # Other pedestrian and future motion.
    other_pos = (0.705, 0.630 if not compact else 0.605)
    add_pedestrian(ax, other_pos, scale=0.45, color="#4e5966", zorder=9)
    if show_other_label:
        ax.text(
            other_pos[0] + 0.020,
            other_pos[1] + 0.080,
            "other agent",
            fontsize=5.0,
            color="#4e5966",
            ha="left",
            va="center",
            zorder=10,
        )
    add_curve(
        ax,
        (other_pos[0] - 0.010, other_pos[1] - 0.020),
        (0.645, 0.500 if not compact else 0.485),
        c1=(0.700, 0.600),
        c2=(0.665, 0.555),
        color="#4e5966",
        lw=0.85,
        alpha=0.70,
        ls=(0, (3, 2)),
        zorder=5,
        end_marker=False,
    )


def add_warning(ax, xy, text, *, color=COLORS["bad"], width=0.145):
    x, y = xy
    ax.add_patch(Circle((x, y), 0.020, fc="white", ec=color, lw=0.8, zorder=13))
    ax.text(x, y - 0.001, "!", ha="center", va="center", fontsize=6.2, color=color, fontweight="bold", zorder=14)
    ax.text(x + 0.028, y, text, ha="left", va="center", fontsize=5.2, color=color, zorder=14)


def add_small_arrow(ax, start, end, color):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            lw=0.85,
            color=color,
            alpha=0.88,
            zorder=12,
        )
    )


def draw_v2_left(ax):
    add_panel_frame(
        ax,
        "a",
        "Best-of-K blind spot",
        "The best match can be correct while many other candidates are implausible.",
    )
    add_scene(ax)
    origin = draw_observed(ax, start=(0.105, 0.300), end=(0.330, 0.405), scale=0.86)

    # Implausible candidates: through obstacle, off-scene, and collision-prone.
    bad_specs = [
        ((0.850, 0.705), (0.470, 0.600), (0.620, 0.760), COLORS["gray"], 0.58),
        ((0.875, 0.545), (0.505, 0.500), (0.670, 0.545), COLORS["bad"], 0.76),
        ((0.795, 0.145), (0.470, 0.260), (0.640, 0.120), COLORS["bad"], 0.74),
        ((0.900, 0.355), (0.520, 0.385), (0.690, 0.340), COLORS["gray"], 0.58),
        ((0.770, 0.845), (0.480, 0.660), (0.650, 0.850), COLORS["gray"], 0.58),
    ]
    for end, c1, c2, color, alpha in bad_specs:
        add_curve(ax, origin, end, c1=c1, c2=c2, color=color, lw=1.05, alpha=alpha, zorder=3)

    gt_end = (0.835, 0.645)
    add_curve(
        ax,
        origin,
        gt_end,
        c1=(0.475, 0.505),
        c2=(0.660, 0.625),
        color=COLORS["gt"],
        lw=1.35,
        alpha=0.95,
        ls=(0, (3, 2)),
        zorder=7,
        end_marker=False,
    )
    ax.plot(gt_end[0], gt_end[1], marker="*", ms=7.5, color=COLORS["gt"], markeredgecolor="white", markeredgewidth=0.5, zorder=9)
    add_curve(
        ax,
        origin,
        (0.810, 0.630),
        c1=(0.475, 0.485),
        c2=(0.650, 0.610),
        color=COLORS["best"],
        lw=2.05,
        alpha=1.0,
        zorder=8,
    )

    add_warning(ax, (0.600, 0.500), "through obstacle")
    add_warning(ax, (0.690, 0.585), "social conflict", color=COLORS["warn"])
    add_warning(ax, (0.680, 0.205), "off feasible area")
    ax.text(0.770, 0.680, "GT", fontsize=5.7, color=COLORS["gt"], ha="right")
    add_chip(ax, (0.475, 0.125), "minADE/minFDE reward only the hit", "#e9f6f8", COLORS["best"], COLORS["best"], width=0.465, fontsize=5.6)


def draw_v2_middle(ax):
    add_panel_frame(
        ax,
        "b",
        "Meaningful diversity",
        "Diversity should cover feasible futures, not random spread or single-GT collapse.",
    )
    add_scene(ax, show_other_label=False)
    origin = draw_observed(ax, start=(0.105, 0.300), end=(0.330, 0.405), scale=0.86)

    feasible_modes = [
        ((0.835, 0.710), (0.465, 0.600), (0.620, 0.745), COLORS["analog"], "yield/left"),
        ((0.860, 0.535), (0.480, 0.490), (0.660, 0.545), COLORS["analog"], "direct"),
        ((0.820, 0.300), (0.465, 0.325), (0.635, 0.280), COLORS["analog"], "right"),
    ]
    for end, c1, c2, color, label in feasible_modes:
        add_curve(
            ax,
            origin,
            end,
            c1=c1,
            c2=c2,
            color=color,
            lw=1.2,
            alpha=0.78,
            ls=(0, (3, 2)),
            zorder=4,
        )
        label_y = end[1] - 0.010 if label == "yield/left" else end[1]
        ax.text(end[0] + 0.012, label_y, label, fontsize=4.9, color=color, ha="left", va="center", zorder=10)

    # Collapsed candidates around one GT-compatible route.
    for j, off in enumerate([-0.032, -0.018, -0.004, 0.012, 0.028]):
        add_curve(
            ax,
            origin,
            (0.835 + 0.008 * np.cos(j), 0.650 + off * 0.22),
            c1=(0.475, 0.505 + off),
            c2=(0.650, 0.618 + off * 0.3),
            color=COLORS["warn"],
            lw=1.05,
            alpha=0.82,
            zorder=7,
        )
    ax.plot(0.845, 0.655, marker="*", ms=7.0, color=COLORS["gt"], markeredgecolor="white", markeredgewidth=0.5, zorder=9)
    ax.text(0.665, 0.675, "collapsed\nsingle-GT set", fontsize=5.2, color=COLORS["warn"], ha="center", va="bottom")

    add_small_arrow(ax, (0.780, 0.635), (0.805, 0.565), COLORS["bad"])
    ax.text(0.520, 0.215, "plausible modes should remain covered", fontsize=5.6, color=COLORS["analog"], ha="center")
    add_chip(ax, (0.140, 0.125), "accuracy", "#f0edf9", COLORS["refine"], COLORS["refine"], width=0.165, fontsize=5.6)
    add_chip(ax, (0.365, 0.125), "scene plausibility", "#fff4df", COLORS["warn"], COLORS["warn"], width=0.245, fontsize=5.6)
    add_chip(ax, (0.690, 0.125), "mode coverage", "#e8f3fb", COLORS["analog"], COLORS["analog"], width=0.225, fontsize=5.6)


def draw_v2_right(ax):
    add_panel_frame(
        ax,
        "c",
        "Our set-level refinement",
        "Residual refinement improves candidates against scene-aware plausible futures.",
    )
    add_scene(ax, show_other_label=False)
    origin = draw_observed(ax, start=(0.105, 0.300), end=(0.330, 0.405), scale=0.86)

    # Original candidates.
    originals = [
        ((0.850, 0.745), (0.470, 0.615), (0.620, 0.790)),
        ((0.895, 0.565), (0.505, 0.500), (0.690, 0.560)),
        ((0.795, 0.180), (0.470, 0.270), (0.650, 0.125)),
        ((0.905, 0.345), (0.520, 0.385), (0.700, 0.330)),
    ]
    for end, c1, c2 in originals:
        add_curve(ax, origin, end, c1=c1, c2=c2, color=COLORS["gray"], lw=0.9, alpha=0.42, zorder=3)

    analogs = [
        ((0.830, 0.700), (0.465, 0.590), (0.625, 0.720)),
        ((0.865, 0.535), (0.480, 0.490), (0.660, 0.540)),
        ((0.820, 0.305), (0.465, 0.330), (0.635, 0.285)),
    ]
    for end, c1, c2 in analogs:
        add_curve(
            ax,
            origin,
            end,
            c1=c1,
            c2=c2,
            color=COLORS["analog"],
            lw=1.2,
            alpha=0.83,
            ls=(0, (3, 2)),
            zorder=5,
        )

    refined = [
        ((0.815, 0.680), (0.470, 0.570), (0.625, 0.675), COLORS["refine"]),
        ((0.850, 0.530), (0.480, 0.480), (0.655, 0.525), COLORS["refine2"]),
        ((0.805, 0.320), (0.465, 0.340), (0.635, 0.300), COLORS["refine"]),
    ]
    for end, c1, c2, color in refined:
        add_curve(ax, origin, end, c1=c1, c2=c2, color=color, lw=1.90, alpha=1.0, zorder=8)

    add_small_arrow(ax, (0.590, 0.598), (0.635, 0.650), COLORS["refine"])
    add_small_arrow(ax, (0.610, 0.355), (0.660, 0.320), COLORS["refine"])
    ax.text(0.500, 0.735, "analogical future coverage", fontsize=5.4, color=COLORS["analog"], ha="left")
    ax.text(0.395, 0.230, "distribution-level\nresidual correction", fontsize=5.4, color=COLORS["refine"], ha="center")
    add_chip(ax, (0.130, 0.125), "set quality", "#f0edf9", COLORS["refine"], COLORS["refine"], width=0.175, fontsize=5.6)
    add_chip(ax, (0.380, 0.125), "AFC recall", "#e8f3fb", COLORS["analog"], COLORS["analog"], width=0.175, fontsize=5.6)
    add_chip(ax, (0.645, 0.125), "base-mode coverage", "#e9f7f3", COLORS["refine2"], COLORS["refine2"], width=0.270, fontsize=5.6)
    ax.text(0.620, 0.065, "high-quality and plausible diversity", fontsize=6.0, color=COLORS["ink"], fontweight="bold", ha="center")


def draw_top_flow(fig, axes):
    overlay = fig.add_axes([0, 0, 1, 1], zorder=20)
    overlay.axis("off")
    y = 0.905
    for left_ax, right_ax in [(axes[0], axes[1]), (axes[1], axes[2])]:
        b1 = left_ax.get_position()
        b2 = right_ax.get_position()
        overlay.add_patch(
            FancyArrowPatch(
                (b1.x1 + 0.010, y),
                (b2.x0 - 0.010, y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=9,
                lw=0.8,
                color=COLORS["gray_dark"],
                alpha=0.85,
            )
        )


def draw_figure():
    configure_matplotlib()
    width_in = WIDTH_MM / 25.4
    height_in = HEIGHT_MM / 25.4
    fig = plt.figure(figsize=(width_in, height_in))
    fig.text(
        0.5,
        0.965,
        "Beyond Best-of-K: Scene-Aware Quality and Plausible Diversity",
        ha="center",
        va="top",
        fontsize=10.1,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.5,
        0.925,
        "Good multimodal prediction requires accurate, feasible, socially compatible, and diverse future candidates.",
        ha="center",
        va="top",
        fontsize=6.7,
        color=COLORS["muted"],
    )

    axes = [fig.add_axes([left, 0.115, 0.270, 0.755]) for left in [0.045, 0.365, 0.685]]
    draw_v2_left(axes[0])
    draw_v2_middle(axes[1])
    draw_v2_right(axes[2])
    draw_top_flow(fig, axes)
    fig.text(
        0.50,
        0.037,
        "Not all diversity is useful: candidates should cover multiple plausible futures under scene and social constraints.",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["ink"],
        fontweight="bold",
    )
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
