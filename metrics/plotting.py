import numpy as np
import matplotlib.pyplot as plt

from core.utils import combinations
from pathlib import Path


def plot_utilities_2d(u,
                      xlims,
                      ylims,
                      num_actions,
                      title="",
                      cmap="Spectral",
                      save=False,
                      save_dir="",
                      filename="",
                      show_plot=True):
    xmin, xmax = xlims
    ymin, ymax = ylims
    xdomain = np.linspace(xmin, xmax, num_actions)[:, None]
    ydomain = np.linspace(ymin, ymax, num_actions)[:, None]
    combs = combinations(xdomain, ydomain)
    xlabel = 'Agent 1 actions'
    ylabel = 'Agent 2 actions'

    u1_vals = u[0](combs)
    u1_reshaped = np.transpose(
        np.reshape(u1_vals, [num_actions, num_actions]))
    u2_vals = u[1](combs)
    u2_reshaped = np.transpose(
        np.reshape(u2_vals, [num_actions, num_actions]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title, size=20)
    fig.set_size_inches(8, 4)
    fig.set_dpi(200)

    im1 = ax1.imshow(u1_reshaped,
                     interpolation='nearest',
                     extent=(xmin, xmax, ymin, ymax),
                     origin='lower',
                     cmap=cmap,
                     aspect=(xmax - xmin) / (ymax - ymin))
    # ax1.plot(inputs[:, 0], inputs[:, 1], 'ko', mew=2)
    # ax1.plot(*mean_stationary, marker='*', markersize=20, color='white')
    ax1.set_title("Agent 1 utility", size=16)
    ax1.set_xlabel(xlabel, size=12)
    ax1.set_ylabel(ylabel, size=12)
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(u2_reshaped,
                     interpolation='nearest',
                     extent=(xmin, xmax, ymin, ymax),
                     origin='lower',
                     cmap=cmap,
                     aspect=(xmax - xmin) / (ymax - ymin))
    # ax2.plot(inputs[:, 0], inputs[:, 1], 'ko', mew=2)
    # ax2.plot(*f_std_argmin, marker='*', markersize=20, color='white')
    ax2.set_title("Agent 2 utility", size=16)
    ax2.set_xlabel(xlabel, size=12)
    ax2.set_ylabel(ylabel, size=12)
    fig.colorbar(im2, ax=ax2)
    fig.tight_layout()

    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches='tight')

    if show_plot:
        plt.show()
