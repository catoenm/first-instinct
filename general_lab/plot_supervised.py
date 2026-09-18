"""Plot observed held-out accuracy from the saved paired supervised reports."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/general-supervised-v1"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/general-supervised"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    groups = [
        ("test", "public", "Public text · held-out source groups"),
        ("test", "verified", "Generated worlds · familiar reasoning families"),
        ("challenge", "public", "Public text · two held-out skills"),
        ("challenge", "verified", "Generated worlds · new compositions"),
    ]
    rows = []
    for split, family, label in groups:
        report = json.loads((args.results / "reports" / split / "report.json").read_text())
        group = report["general"]["subgroups"][family]
        measured = group["row_weighted"]
        rows.append((f"{label}\n{group['rows']:,} questions", measured["base"]["accuracy"], measured["trained"]["accuracy"]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(11.4, 5.1))
    foundation, adapted = "#84949f", "#087e91"
    for y, (_, before, after) in enumerate(rows):
        axis.plot([before * 100, after * 100], [y, y], color="#dce4e9", linewidth=5, zorder=1)
        axis.scatter(before * 100, y, s=90, color=foundation, label="Untouched foundation" if y == 0 else None, zorder=2)
        axis.scatter(after * 100, y, s=90, color=adapted, label="Supervised adaptation" if y == 0 else None, zorder=3)
        axis.annotate(f"{before:.1%}", (before * 100, y), xytext=(0, -20), textcoords="offset points", ha="center", color="#53636e", fontsize=10)
        axis.annotate(f"{after:.1%}", (after * 100, y), xytext=(0, 10), textcoords="offset points", ha="center", color=adapted, fontsize=11, weight="bold")
    axis.set_yticks(range(len(rows)), [row[0] for row in rows])
    axis.invert_yaxis()
    axis.set_ylim(len(rows) - .45, -.65)
    axis.set_xlim(0, 100)
    axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.set_xlabel("Accuracy · each question has equal weight", labelpad=10)
    axis.grid(axis="x", color="#edf0f3")
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=0, pad=9)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.suptitle("One supervised pass improves constrained decisions", x=.045, y=.98, ha="left", fontsize=18, weight="bold", color="#152a3a")
    figure.text(.045, .91, "Qwen3.5-9B · identical prompts and answer scoring · 350,857 training examples", color="#53636e", fontsize=11)
    axis.legend(loc="lower right", bbox_to_anchor=(1, 1.11), frameon=False, ncol=2, fontsize=10, handletextpad=.3, columnspacing=1)
    figure.text(.045, .025, "One training seed. Public pretraining exposure is unknown. On 32 prose pairs, both-correct accuracy stayed at 26/32.", fontsize=9, color="#53636e")
    figure.subplots_adjust(left=.42, right=.96, top=.77, bottom=.18)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output / f"held-out-accuracy.{extension}"
        figure.savefig(path, dpi=180, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
