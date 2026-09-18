"""Plot selected or latest reinforcement checkpoints from a verified paired report.

Reads saved measurements only. Each interval resamples complete evaluation
worlds within one trained run; training seeds are displayed separately.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/general-reinforcement"))
    parser.add_argument("--checkpoint", choices=("best", "latest"), default="best")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    runs = sorted(report["reinforcement_learning"]["runs"],
                  key=lambda run: (run["config"]["seed"], run["config"]["forecast_weight"]))
    if not runs or any(run["status"] != "complete" for run in runs):
        raise ValueError("The figure requires completed reinforcement runs")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    splits = [("test", "Ordinary holdout", "#087e91", -.19),
              ("shift", "Changed conditions", "#a25625", 0),
              ("new_domain", "Reserved wording", "#7657a0", .19)]
    metrics = [("policy_expected_reward", "Change in expected reward", "Higher is better"),
               ("posterior_mean_squared_error", "Change in squared probability error", "Lower is better")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    figure, axes = plt.subplots(1, 2, figsize=(12, max(4.7, 1.0 * len(runs) + 1.9)), sharey=True)
    labels = []
    for index, run in enumerate(runs):
        method = "Reward + forecast" if run["config"]["forecast_weight"] else "Reward"
        update = run["selected_update"] if args.checkpoint == "best" else run["updates"]
        labels.append(f"{method} · seed {run['config']['seed']}\n{'Selected' if args.checkpoint == 'best' else 'Latest'} update {update}")
        for split, label, color, offset in splits:
            comparison = run["paired_baseline_comparisons"][f"baseline_to_{args.checkpoint}:{split}"]
            if not comparison["identity_verified"]:
                raise ValueError("The figure requires paired world identities")
            for axis, (metric, _, _) in zip(axes, metrics):
                interval = comparison["paired_world_bootstrap"][metric]
                delta, low, high = (interval[key] for key in ("difference", "lower_95", "upper_95"))
                y = index + offset
                axis.plot([low, high], [y, y], color=color, linewidth=1.7)
                axis.scatter([delta], [y], color=color, s=32,
                             label=label if index == 0 and axis is axes[0] else None, zorder=3)
    for axis, (_, title, direction) in zip(axes, metrics):
        axis.axvline(0, color="#6d7980", linewidth=.8)
        axis.set_title(f"{title}\n{direction}", fontsize=11, pad=18)
        axis.grid(axis="x", color="#edf0f3")
        axis.set_axisbelow(True)
        axis.tick_params(length=0, pad=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlabel(f"{'Selected' if args.checkpoint == 'best' else 'Latest'} checkpoint minus supervised starting model", fontsize=9, labelpad=10)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].set_yticks(range(len(runs)), labels)
    axes[0].set_ylim(len(runs) - .45, -.55)
    figure.suptitle("Reward and forecast changes after reinforcement learning", x=.035, y=.98,
                   ha="left", fontsize=18, weight="bold", color="#152a3a")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper left",
                  bbox_to_anchor=(.027, .93), ncol=3, frameon=False, fontsize=10)
    figure.text(.035, .04, "Intervals: 95% paired world bootstrap. Seeds are separate; intervals exclude training-seed uncertainty.",
                fontsize=9, color="#53636e")
    figure.text(.035, .012, "Probability error compares forecasts with the simulator's exact posterior. One shared environment mechanism.",
                fontsize=9, color="#53636e")
    figure.subplots_adjust(left=.255, right=.975, top=.73, bottom=.19, wspace=.30)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output / f"{args.checkpoint}-reward-and-forecast.{extension}"
        figure.savefig(path, dpi=180, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


if __name__ == "__main__":
    main()
