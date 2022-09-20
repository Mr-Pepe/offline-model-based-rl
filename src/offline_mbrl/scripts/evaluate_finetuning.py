import d4rl
import matplotlib.pyplot as plt
import numpy as np

from offline_mbrl.evaluation.plot import get_all_datasets, plot_data

if __name__ == "__main__":
    # A list of lists
    # Each sublist contains paths to experiment directories that contain a progress.txt
    # Experiments from each sublist are plotted together
    logdirs: list[list[str]] = []

    env_names = [
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "walker2d-medium-v2",
    ]

    legend = [
        "aleatoric penalty",
        "aleatoric partitioning",
        "epistemic penalty",
        "epistemic partitioning",
    ]

    titles = ["Halfcheetah", "Hopper", "Walker2d"]

    f, axes = plt.subplots(1, 3, figsize=(8, 3.5))

    for i_axis, ax in enumerate(axes):

        def normalize_score(x, i_env):
            return d4rl.get_normalized_score(env_names[i_env], x) * 100

        plt.sca(ax)
        data = get_all_datasets(logdirs[i_axis], legend=legend)
        for datum in data:
            datum["AverageTestEpRet"] = [
                normalize_score(x, i_axis) for x in datum["AverageTestEpRet"]
            ]
        condition = "Condition1"
        estimator = getattr(np, "mean")
        plot_data(
            data,
            xaxis="Epoch",
            value="AverageTestEpRet",
            condition=condition,
            smooth=15,
            estimator=estimator,
        )
        ax.grid(b=True, alpha=0.5, linestyle="--")
        ax.get_legend().remove()
        ax.set_title(titles[i_axis], fontsize=14)
        ax.set_xticks([0, 50, 100])
        ax.set_ylabel(None)

    axes[0].set_ylabel("Performance")

    if (
        axes[0].get_legend_handles_labels()[1] != axes[1].get_legend_handles_labels()[1]
        or axes[0].get_legend_handles_labels()[1]
        != axes[2].get_legend_handles_labels()[1]
    ):
        raise AssertionError("Legend handles not identical")

    f.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        prop={"size": 12},
        ncol=2
    )

    f.subplots_adjust(
        top=0.919, bottom=0.314, left=0.086, right=0.977, hspace=0.2, wspace=0.291
    )
    plt.show()
