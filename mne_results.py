import pickle
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from metrics.plotting import smooth_curve

ex = Experiment("NashBO_results")
ex.observers.append(FileStorageObserver("../runs"))


@ex.named_config
def rand():
    utility_name = "rand"
    num_bo_iters = 200
    num_seeds = 5


@ex.named_config
def gan():
    utility_name = "gan"
    num_bo_iters = 400
    num_seeds = 5


@ex.named_config
def bcad():
    utility_name = "bcad"
    num_bo_iters = 400
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
    base_dir = "results/mne/" + utility_name + "/"
    save_dir = base_dir
    pickles_dir = base_dir + "pickles/"

    acquisitions = ["random_mne", "max_ent_mne", "ucb_mne_noexplore", "ucb_mne"]
    color_dict = {
        "ucb_mne_noexplore": "#26c485",
        "max_ent_mne": "#fbb13c",
        "ucb_mne": "#00a6ed",
        "random_mne": "#d7263d",
    }
    acq_name_dict = {
        "ucb_mne_noexplore": "UCB-MNE no-exp",
        "max_ent_mne": "Max entropy",
        "ucb_mne": "UCB-MNE (ours)",
        "random_mne": "Random",
    }

    # Plot simple regret
    fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    for acquisition in acquisitions:
        color = color_dict[acquisition]

        if utility_name in ["rand", "gan"]:
            cutoff = num_bo_iters
        else:
            cutoff = 200

        all_simple_regrets = np.zeros((num_seeds, cutoff))
        all_times = []
        for seed in range(num_seeds):
            filename = f"mne-{utility_name}-{acquisition}-seed{seed}.p"
            (
                reported_strategies,
                sampled_strategies,
                sample_regret,
                cumu_regret,
                time_per_iter,
                args,
            ) = pickle.load(open(pickles_dir + filename, "rb"))

            all_simple_regrets[seed] = np.minimum.accumulate(sample_regret)[:cutoff]
            all_times.append(time_per_iter)
        mean_simple_regrets = np.mean(all_simple_regrets, axis=0)
        std_err_simple_regrets = np.std(all_simple_regrets, axis=0) / np.sqrt(num_seeds)
        acq_name = acq_name_dict[acquisition]

        x = np.arange(cutoff)
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
        axs.set_ylabel("Simple mixed Nash regret", size=text_size)
        axs.tick_params(labelsize=tick_size)
        axs.legend(fontsize=text_size - 2, loc="upper right")
        axs.set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        save_dir + f"mne-{utility_name}-simple_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )
