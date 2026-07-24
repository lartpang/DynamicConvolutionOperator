"""Generate the README optimization-progression chart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


STAGES = ("Unfold\nbaseline", "Triton\ndynamic", "Generator\nfused")
SERIES = {
    "128² FP32": {
        "color": "#2563eb",
        "marker": "o",
        "linestyle": "-",
        "inference": (1.0, 1.814, 2.782),
        "training": (1.0, 1.113, 1.178),
    },
    "128² BF16": {
        "color": "#ea580c",
        "marker": "s",
        "linestyle": "--",
        "inference": (1.0, 2.239, 3.723),
        "training": (1.0, 1.065, 1.045),
    },
    "256² FP32": {
        "color": "#16a34a",
        "marker": "^",
        "linestyle": "-",
        "inference": (1.0, 1.874, 3.726),
        "training": (1.0, 2.082, 2.162),
    },
    "256² BF16": {
        "color": "#9333ea",
        "marker": "D",
        "linestyle": "--",
        "inference": (1.0, 2.016, 5.226),
        "training": (1.0, 2.251, 2.783),
    },
}


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#64748b",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    figure.suptitle(
        "DDPM end-to-end speedup across optimization stages · RTX 5090",
        fontsize=15,
        fontweight="bold",
    )

    for axis, metric, title, upper in (
        (axes[0], "inference", "Inference", 5.65),
        (axes[1], "training", "Training", 3.05),
    ):
        for label, config in SERIES.items():
            values = config[metric]
            axis.plot(
                STAGES,
                values,
                color=config["color"],
                marker=config["marker"],
                linestyle=config["linestyle"],
                linewidth=2.2,
                markersize=7,
                label=label,
            )
            for index, value in enumerate(values):
                if index == 0:
                    continue
                axis.annotate(
                    f"{value:.2f}×",
                    (index, value),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    color=config["color"],
                    fontsize=8,
                    fontweight="bold",
                )
        axis.axhline(1.0, color="#94a3b8", linewidth=1, linestyle=":")
        axis.set_title(title, pad=10)
        axis.set_ylabel("Speedup vs. Unfold (×)")
        axis.set_ylim(0.8, upper)
        axis.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=False,
    )
    figure.text(
        0.5,
        0.015,
        "Training chart shows the forced fused path; automatic routing keeps the faster "
        "materialized result for 128² BF16.",
        ha="center",
        color="#475569",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.84))
    output_path = Path(__file__).with_name("performance_progression.svg")
    figure.savefig(
        output_path,
        format="svg",
        facecolor="#ffffff",
        transparent=False,
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "plot_performance.py"},
    )
    svg = output_path.read_text(encoding="utf-8")
    output_path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
