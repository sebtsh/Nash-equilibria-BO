import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("NashBO_results")
ex.observers.append(FileStorageObserver("../runs"))


@ex.named_config
def rand():
    utility_name = "rand"
    num_bo_iters = 200
    num_seeds = 4


@ex.named_config
def gan():
    utility_name = "gan"
    num_bo_iters = 600
    num_seeds = 5


@ex.named_config
def bcad():
    utility_name = "bcad"
    num_bo_iters = 600
    num_seeds = 5


@ex.automain
def main(
    utility_name,
    num_bo_iters,
    num_seeds,
    figsize=(10, 6),
    dpi=200,
):
    text_size = 28
    tick_size = 20
    base_dir = "results/pne/" + utility_name + "/"
    save_dir = base_dir
    pickles_dir = base_dir + "pickles/"

    acquisitions = ["ucb_pne", "prob_eq", "BN"]
    x = np.arange(num_bo_iters)
    color_dict = {
        "prob_eq": "#d7263d",
        "BN": "#fbb13c",
        "ucb_pne": "#00a6ed",
    }
    acq_name_dict = {
        "prob_eq": "Prob. Equil.",
        "BN": "BN",
        "ucb_pne": "UCB-PNE",
    }

    # Plot cumulative regret
    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    for acquisition in acquisitions:
        color = color_dict[acquisition]
        all_cumu_regrets = np.zeros((num_seeds, num_bo_iters))
        all_times = []
        for seed in range(num_seeds):
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

            all_cumu_regrets[seed] = reported_cumu_regret
            all_times.append(time_per_iter)
        mean_cumu_regrets = np.mean(all_cumu_regrets, axis=0)
        std_err_cumu_regrets = np.std(all_cumu_regrets, axis=0) / np.sqrt(num_seeds)
        acq_name = acq_name_dict[acquisition]

        # Cumulative regret
        axs.plot(x, mean_cumu_regrets, label=acq_name, color=color)
        axs.fill_between(
            x,
            mean_cumu_regrets - std_err_cumu_regrets,
            mean_cumu_regrets + std_err_cumu_regrets,
            alpha=0.2,
            color=color,
        )
        # axs[i].legend(fontsize=20)
        axs.set_xlabel("Iterations", size=text_size)
        axs.set_ylabel("Cumu. pure Nash regret", size=text_size)
        axs.tick_params(labelsize=tick_size)
        axs.legend(fontsize=text_size-2, loc='lower left')

    fig.tight_layout()
    fig.savefig(
        save_dir + f"pne-{utility_name}-cumu_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )

    # Plot immediate regret
    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    axs.set_yscale("log")

    for acquisition in acquisitions:
        color = color_dict[acquisition]
        all_imm_regrets = np.zeros((num_seeds, num_bo_iters))
        all_times = []
        for seed in range(num_seeds):
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

            all_imm_regrets[seed] = reported_sample_regret
            all_times.append(time_per_iter)
        mean_imm_regrets = np.mean(all_imm_regrets, axis=0)
        std_err_imm_regrets = np.std(all_imm_regrets, axis=0) / np.sqrt(num_seeds)
        acq_name = acq_name_dict[acquisition]

        # Cumulative regret
        axs.plot(x, mean_imm_regrets, label=acq_name, color=color)
        axs.fill_between(
            x,
            mean_imm_regrets - std_err_imm_regrets,
            mean_imm_regrets + std_err_imm_regrets,
            alpha=0.2,
            color=color,
        )
        # axs[i].legend(fontsize=20)
        axs.set_xlabel("Iterations", size=text_size)
        axs.set_ylabel("Imm. pure Nash regret", size=text_size)
        axs.tick_params(labelsize=tick_size)
        axs.legend(fontsize=text_size-2, loc='lower left')

    fig.tight_layout()
    fig.savefig(
        save_dir + f"pne-{utility_name}-imm_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )

    # Plot sample regret
    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    axs.set_yscale("log")

    for acquisition in acquisitions:
        color = color_dict[acquisition]
        all_sample_regrets = np.zeros((num_seeds, num_bo_iters))
        all_times = []
        for seed in range(num_seeds):
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

            all_sample_regrets[seed] = sampled_sample_regret
            all_times.append(time_per_iter)
        mean_sample_regrets = np.mean(all_sample_regrets, axis=0)
        std_err_sample_regrets = np.std(all_sample_regrets, axis=0) / np.sqrt(num_seeds)
        acq_name = acq_name_dict[acquisition]

        # Cumulative regret
        axs.plot(x, mean_sample_regrets, label=acq_name, color=color)
        axs.fill_between(
            x,
            mean_sample_regrets - std_err_sample_regrets,
            mean_sample_regrets + std_err_sample_regrets,
            alpha=0.2,
            color=color,
        )
        # axs[i].legend(fontsize=20)
        axs.set_xlabel("Iterations", size=text_size)
        axs.set_ylabel("Sample pure Nash regret", size=text_size)
        axs.tick_params(labelsize=tick_size)
        axs.legend(fontsize=text_size-2, loc='lower left')

    fig.tight_layout()
    fig.savefig(
        save_dir + f"pne-{utility_name}-sample_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )
