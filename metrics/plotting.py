import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.pne import best_response_payoff_pure_discrete
from core.utils import cross_product
from core.models import create_ci_funcs


def plot_utilities_2d(
    u,
    bounds,
    num_discrete=500,
    title="",
    cmap="Spectral",
    save=False,
    save_dir="",
    filename="",
    show_plot=True,
    known_best_point=None,
):
    ymin, ymax = bounds[0]
    xmin, xmax = bounds[1]

    actions1 = np.linspace(ymin, ymax, num_discrete)
    actions2 = np.linspace(xmin, xmax, num_discrete)
    domain = cross_product(actions1[:, None], actions2[:, None])

    xlabel = "Agent 2 actions"
    ylabel = "Agent 1 actions"
    # print(f"domain: {np.reshape(domain, [num_discrete, num_discrete, 2])}")
    u1_vals = u[0](domain)
    u1_reshaped = np.reshape(u1_vals, [num_discrete, num_discrete])
    # print(f"u1: {u1_reshaped}")
    u2_vals = u[1](domain)
    u2_reshaped = np.reshape(u2_vals, [num_discrete, num_discrete])
    # print(f"u2: {u2_reshaped}")

    offset = (1 / num_discrete) / 2
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title, size=20)
    fig.set_size_inches(8, 4)
    fig.set_dpi(200)

    im1 = ax1.imshow(
        u1_reshaped,
        interpolation="nearest",
        extent=(ymin - offset, ymax + offset, xmax + offset, xmin - offset),
        origin="upper",
        cmap=cmap,
        # aspect=(ymax - ymin) / (xmax - xmin),
    )
    ax1.set_title("Agent 1 utility", size=16)
    ax1.set_xlabel(xlabel, size=12)
    ax1.set_ylabel(ylabel, size=12)
    fig.colorbar(im1, ax=ax1)
    if known_best_point is not None:
        ax1.plot(
            known_best_point[:, 1],
            known_best_point[:, 0],
            "*",
            markersize=10,
            c="black",
        )
        ax1.axvline(
            x=known_best_point[:, 1],
            ymin=ymin,
            ymax=ymax,
            c="black",
            linestyle="--",
            alpha=0.5,
        )

    im2 = ax2.imshow(
        u2_reshaped,
        interpolation="None",
        extent=(ymin - offset, ymax + offset, xmax + offset, xmin - offset),
        origin="upper",
        cmap=cmap,
        # aspect=(ymax - ymin) / (xmax - xmin),
    )
    ax2.set_title("Agent 2 utility", size=16)
    ax2.set_xlabel(xlabel, size=12)
    ax2.set_ylabel(ylabel, size=12)
    fig.colorbar(im2, ax=ax2)
    if known_best_point is not None:
        ax2.plot(
            known_best_point[:, 1],
            known_best_point[:, 0],
            "*",
            markersize=10,
            c="black",
        )
        ax2.axhline(
            y=known_best_point[:, 0],
            xmin=xmin,
            xmax=xmax,
            c="black",
            linestyle="--",
            alpha=0.5,
        )

    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_fi_2d(
    models,
    beta,
    bounds,
    num_discrete=200,
    title="",
    cmap="Spectral",
    save=False,
    save_dir="",
    filename="",
    show_plot=True,
    known_best_point=None,
):
    ucb_funcs, lcb_funcs = create_ci_funcs(models, beta)
    ymin, ymax = bounds[0]
    xmin, xmax = bounds[1]

    actions1 = np.linspace(ymin, ymax, num_discrete)
    actions2 = np.linspace(xmin, xmax, num_discrete)
    domain = cross_product(actions1[:, None], actions2[:, None])

    # Calculate f_i for agent 1 only
    xlabel = "Agent 2 actions"
    ylabel = "Agent 1 actions"
    # print(f"domain: {np.reshape(domain, [num_discrete, num_discrete, 2])}")
    ucb_vals = ucb_funcs[0](domain)
    ucb_reshaped = np.reshape(ucb_vals, [num_discrete, num_discrete])
    # print(f"u1: {u1_reshaped}")
    lcb_vals = lcb_funcs[0](domain)
    lcb_reshaped = np.reshape(lcb_vals, [num_discrete, num_discrete])
    # print(f"u2: {u2_reshaped}")
    f1_reshaped = ucb_reshaped - np.max(lcb_reshaped, axis=0, keepdims=True)

    offset = (1 / num_discrete) / 2
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title, size=20)
    fig.set_size_inches(8, 4)
    fig.set_dpi(200)

    im1 = ax1.imshow(
        f1_reshaped,
        interpolation="nearest",
        extent=(ymin - offset, ymax + offset, xmax + offset, xmin - offset),
        origin="upper",
        cmap=cmap,
        # aspect=(ymax - ymin) / (xmax - xmin),
    )
    ax1.set_title("$\\widehat f_{1, t-1}$", size=16)
    ax1.set_xlabel(xlabel, size=12)
    ax1.set_ylabel(ylabel, size=12)
    fig.colorbar(im1, ax=ax1)
    if known_best_point is not None:
        ax1.plot(
            known_best_point[:, 1],
            known_best_point[:, 0],
            "*",
            markersize=10,
            c="black",
        )
        ax1.axvline(
            x=known_best_point[:, 1],
            ymin=ymin,
            ymax=ymax,
            c="black",
            linestyle="--",
            alpha=0.5,
        )

    im2 = ax2.imshow(
        ucb_reshaped,
        interpolation="None",
        extent=(ymin - offset, ymax + offset, xmax + offset, xmin - offset),
        origin="upper",
        cmap=cmap,
        # aspect=(ymax - ymin) / (xmax - xmin),
    )
    ax2.set_title("\\widehat u_{1, t-1}", size=16)
    ax2.set_xlabel(xlabel, size=12)
    ax2.set_ylabel(ylabel, size=12)
    fig.colorbar(im2, ax=ax2)
    if known_best_point is not None:
        ax2.plot(
            known_best_point[:, 1],
            known_best_point[:, 0],
            "*",
            markersize=10,
            c="black",
        )
        ax2.axhline(
            y=known_best_point[:, 0],
            xmin=xmin,
            xmax=xmax,
            c="black",
            linestyle="--",
            alpha=0.5,
        )

    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_utilities_2d_discrete(
    u,
    xlims,
    ylims,
    actions,
    domain,
    response_dicts,
    title="",
    cmap="Spectral",
    save=False,
    save_dir="",
    filename="",
    show_plot=True,
):
    xmin, xmax = xlims
    ymin, ymax = ylims
    num_actions = len(actions)
    xlabel = "Agent 2 actions"
    ylabel = "Agent 1 actions"

    print(f"domain: {np.reshape(domain, [num_actions, num_actions, 2])}")
    u1_vals = u[0](domain)
    u1_reshaped = np.reshape(u1_vals, [num_actions, num_actions])
    print(f"u1: {u1_reshaped}")
    u2_vals = u[1](domain)
    u2_reshaped = np.reshape(u2_vals, [num_actions, num_actions])
    print(f"u2: {u2_reshaped}")

    brp = best_response_payoff_pure_discrete(
        u=u, domain=domain, response_dicts=response_dicts
    )  # array of shape (M ** N, N)
    nne_idx = np.argmin(np.max(brp, axis=-1))
    nne = domain[nne_idx : nne_idx + 1]
    print(nne)

    offset = (1 / num_actions) / 2
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title, size=20)
    fig.set_size_inches(8, 4)
    fig.set_dpi(200)

    im1 = ax1.imshow(
        u1_reshaped,
        interpolation="nearest",
        extent=(ymin - offset, ymax + offset, xmax + offset, xmin - offset),
        origin="upper",
        cmap=cmap,
        # aspect=(ymax - ymin) / (xmax - xmin),
    )
    ax1.set_title("Agent 1 utility", size=16)
    ax1.set_xlabel(xlabel, size=12)
    ax1.set_ylabel(ylabel, size=12)
    ax1.plot(nne[:, 1], nne[:, 0], "*", markersize=10, c="white")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(
        u2_reshaped,
        interpolation="None",
        extent=(ymin - offset, ymax + offset, xmax + offset, xmin - offset),
        origin="upper",
        cmap=cmap,
        # aspect=(ymax - ymin) / (xmax - xmin),
    )
    ax2.set_title("Agent 2 utility", size=16)
    ax2.set_xlabel(xlabel, size=12)
    ax2.set_ylabel(ylabel, size=12)
    ax2.plot(nne[:, 1], nne[:, 0], "*", markersize=10, c="white")
    fig.colorbar(im2, ax=ax2)

    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_func_2d(
    f,
    xlims,
    ylims,
    actions,
    title="",
    cmap="Spectral",
    save=False,
    save_dir="",
    filename="",
    show_plot=True,
):
    xmin, xmax = xlims
    ymin, ymax = ylims
    num_actions = len(actions)
    xlabel = "Agent 2 actions"
    ylabel = "Agent 1 actions"

    f_reshaped = np.reshape(f, [num_actions, num_actions])
    # f needs to have shape (num_actions ** 2) or (num_actions ** 2, 1)

    offset = (1 / num_actions) / 2
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle(title, size=12)
    fig.set_size_inches(6, 3)
    fig.set_dpi(200)

    im1 = ax1.imshow(
        f_reshaped,
        interpolation="None",
        extent=(ymin - offset, ymax + offset, xmax + offset, xmin - offset),
        origin="upper",
        cmap=cmap,
        # aspect=(xmax - xmin) / (ymax - ymin),
    )
    ax1.set_xlabel(xlabel, size=8)
    ax1.set_ylabel(ylabel, size=8)
    fig.colorbar(im1, ax=ax1)

    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_models_2d(
    models,
    xlims,
    ylims,
    actions,
    domain,
    X=None,
    title="",
    cmap="Spectral",
    save=False,
    save_dir="",
    filename="",
    show_plot=True,
):
    xmin, xmax = xlims
    ymin, ymax = ylims
    num_actions = len(actions)
    xlabel = "Agent 2 actions"
    ylabel = "Agent 1 actions"

    g1_mean, g1_var = models[0].predict_f(domain)
    g1_mean_reshaped = np.reshape(g1_mean, [num_actions, num_actions])
    g1_var_reshaped = np.reshape(g1_var, [num_actions, num_actions])
    g2_mean, g2_var = models[1].predict_f(domain)
    g2_mean_reshaped = np.reshape(g2_mean, [num_actions, num_actions])
    g2_var_reshaped = np.reshape(g2_var, [num_actions, num_actions])

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title, size=10)
    fig.set_size_inches(5, 4)
    fig.set_dpi(200)

    offset = (1 / num_actions) / 2
    for i in range(2):
        for j in range(2):
            if i == 0:
                if j == 0:
                    f = g1_mean_reshaped
                    title = "Agent 1 GP mean"
                elif j == 1:
                    f = g2_mean_reshaped
                    title = "Agent 2 GP mean"
            elif i == 1:
                if j == 0:
                    f = g1_var_reshaped
                    title = "Agent 1 GP variance"
                elif j == 1:
                    f = g2_var_reshaped
                    title = "Agent 2 GP variance"
            ax = axs[i, j]
            im = ax.imshow(
                f,
                interpolation="None",
                extent=(ymin - offset, ymax + offset, xmax + offset, xmin - offset),
                origin="upper",
                cmap=cmap,
                # aspect=(xmax - xmin) / (ymax - ymin),
            )
            if X is not None:
                ax.plot(X[:, 0], X[:, 1], "x", mew=1, c="black")
            ax.set_title(title, size=8)
            ax.set_xlabel(xlabel, size=8)
            ax.set_ylabel(ylabel, size=8)
            fig.colorbar(im, ax=ax)

    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_regret(
    regret, num_iters, title="", save=False, save_dir="", filename="", show_plot=False
):
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle(title, size=12)
    fig.set_size_inches(12, 6)
    fig.set_dpi(200)

    ax1.plot(np.arange(num_iters), regret)
    ax1.axhline(y=0, xmax=num_iters, color="grey", alpha=0.5, linestyle="--")

    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()


def plot_imm_regret(
    regret,
    num_iters,
    skip_length,
    title="",
    save=False,
    save_dir="",
    filename="",
    show_plot=False,
):
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle(title, size=12)
    fig.set_size_inches(12, 6)
    fig.set_dpi(200)

    ax1.plot(np.arange(0, num_iters, skip_length), regret)
    ax1.axhline(y=0, xmax=num_iters, color="grey", alpha=0.5, linestyle="--")

    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()
