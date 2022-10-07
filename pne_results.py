import pickle
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("NashBO_results")
ex.observers.append(FileStorageObserver("../runs"))


@ex.named_config
def rand():
    utility_name = "rand"
    num_bo_iters = 200
    num_seeds = 5
    legend_loc = "lower left"


@ex.named_config
def gan():
    utility_name = "gan"
    num_bo_iters = 600
    num_seeds = 5
    legend_loc = "upper right"


@ex.named_config
def bcad():
    utility_name = "bcad"
    num_bo_iters = 600
    num_seeds = 5
    legend_loc = "upper right"


@ex.automain
def main(
    utility_name,
    num_bo_iters,
    num_seeds,
    legend_loc,
    figsize=(10, 6),
    dpi=200,
):
    text_size = 28
    tick_size = 20
    base_dir = "results/pne/" + utility_name + "/"
    save_dir = base_dir
    pickles_dir = base_dir + "pickles/"

    acquisitions = ["prob_eq", "BN", "ucb_pne_noexplore", "ucb_pne"]
    x = np.arange(num_bo_iters)
    color_dict = {
        "prob_eq": "#d7263d",
        "BN": "#fbb13c",
        "ucb_pne": "#00a6ed",
        "ucb_pne_noexplore": "#26c485",
    }
    acq_name_dict = {
        "prob_eq": "PE",
        "BN": "BN",
        "ucb_pne": "UCB-PNE (ours)",
        "ucb_pne_noexplore": "UCB-PNE no-exp",
    }

    if utility_name == "rand":
        seeds = [4, 19, 20, 70, 102]  # seeds with non-zero epsilon
    else:
        seeds = list(range(num_seeds))

    # Plot simple regret of reported strategy profiles
    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    for acquisition in acquisitions:
        color = color_dict[acquisition]
        all_simple_regrets = np.zeros((num_seeds, num_bo_iters))
        all_times = []
        for i, seed in enumerate(seeds):
            filename = f"pne-{utility_name}-{acquisition}-seed{seed}-2.p"
            (
                reported_strategies,
                sampled_strategies,
                reported_sample_regret,
                reported_cumu_regret,
                sampled_sample_regret,
                sampled_cumu_regret,
                time_per_iter,
                args,
            ) = pickle.load(open(pickles_dir + filename, "rb"))

            all_simple_regrets[i] = np.minimum.accumulate(reported_sample_regret)
            all_times.append(time_per_iter)
        mean_simple_regrets = np.mean(all_simple_regrets, axis=0)
        std_err_simple_regrets = np.std(all_simple_regrets, axis=0) / np.sqrt(num_seeds)
        acq_name = acq_name_dict[acquisition]

        # Cumulative regret
        axs.plot(x, mean_simple_regrets, label=acq_name, color=color)
        axs.fill_between(
            x,
            mean_simple_regrets - std_err_simple_regrets,
            mean_simple_regrets + std_err_simple_regrets,
            alpha=0.2,
            color=color,
        )
        # axs[i].legend(fontsize=20)
        axs.set_xlabel("Iterations", size=text_size)
        axs.set_ylabel("Simple pure Nash regret", size=text_size)
        axs.tick_params(labelsize=tick_size)
        axs.legend(fontsize=text_size - 2, loc=legend_loc)
        axs.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        save_dir + f"pne-{utility_name}-reported_simple_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )

    # Plot simple regret of reported strategy profiles
    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    for acquisition in acquisitions:
        color = color_dict[acquisition]
        all_simple_regrets = np.zeros((num_seeds, num_bo_iters))
        all_times = []
        for i, seed in enumerate(seeds):
            filename = f"pne-{utility_name}-{acquisition}-seed{seed}-2.p"
            (
                reported_strategies,
                sampled_strategies,
                reported_sample_regret,
                reported_cumu_regret,
                sampled_sample_regret,
                sampled_cumu_regret,
                time_per_iter,
                args,
            ) = pickle.load(open(pickles_dir + filename, "rb"))

            all_simple_regrets[i] = np.minimum.accumulate(sampled_sample_regret)
            all_times.append(time_per_iter)
        mean_simple_regrets = np.mean(all_simple_regrets, axis=0)
        std_err_simple_regrets = np.std(all_simple_regrets, axis=0) / np.sqrt(num_seeds)
        acq_name = acq_name_dict[acquisition]

        # Cumulative regret
        axs.plot(x, mean_simple_regrets, label=acq_name, color=color)
        axs.fill_between(
            x,
            mean_simple_regrets - std_err_simple_regrets,
            mean_simple_regrets + std_err_simple_regrets,
            alpha=0.2,
            color=color,
        )
        # axs[i].legend(fontsize=20)
        axs.set_xlabel("Iterations", size=text_size)
        axs.set_ylabel("Simple pure Nash regret", size=text_size)
        axs.tick_params(labelsize=tick_size)
        axs.legend(fontsize=text_size - 2, loc=legend_loc)
        axs.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        save_dir + f"pne-{utility_name}-sampled_simple_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )
