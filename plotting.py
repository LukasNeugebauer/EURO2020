"""
Functions to plot predictions and possibly model fit
"""


import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from itertools import product
import numpy as np
from copy import deepcopy
from collections import defaultdict


def plot_goal_predictions(A, home_team, away_team, ax=None, odds=False):
    """
    Plot the posterior probability of exact outcomes
    """
    odd_values = deepcopy(A)
    odd_values = (1 / odd_values).round(2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    sns.heatmap(A, cmap="magma", cbar=False, ax=ax)
    x_max, y_max = A.shape
    for x, y in product(range(x_max), range(y_max)):
        if odds:
            txt = ax.text(
                y + 0.5,
                x + 0.5,
                str(odd_values[x, y]),
                ha="center",
                va="center",
                color="w",
                size=20,
            )
        else:
            txt = ax.text(
                y + 0.5,
                x + 0.5,
                str(round(A[x, y] * 100, 2)) + " %",
                ha="center",
                va="center",
                color="w",
                size=20,
            )
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground="k")])
    ax.set_aspect("equal")
    ax.set_xlabel(away_team, fontsize=30)
    ax.set_ylabel(home_team, fontsize=30)
    ax.set_xticks(np.arange(A.shape[1]) + 0.5)
    ax.set_xticklabels(np.arange(A.shape[1]), fontsize=25)
    ax.set_yticks(np.arange(A.shape[0]) + 0.5)
    ax.set_yticklabels(np.arange(A.shape[0]), fontsize=25)
    ax.set_title("Exact predictions", fontsize=30)
    if ax is None:
        return fig


def plot_outcome_prob(A, home_team, away_team, ax=None, odds=False):
    """
    Plot the winning and draw probabilities
    """
    p_home = np.tril(A, -1).sum()
    p_away = np.triu(A, 1).sum()
    p_draw = np.diag(A).sum()
    x = np.arange(3)
    y = np.array([p_home, p_draw, p_away])
    odd_values = (1 / y).round(2)
    if ax is None:
        ax = plt.axes()
    sns.barplot(x=x, y=y, hue=y, palette="magma", dodge=False, ax=ax)
    ax.legend("")
    ax.set_xticks(x)
    ax.set_yticks([])
    ax.set_xticklabels([home_team, "Draw", away_team], fontsize=25)
    ax.set_title("Outcome probabilities", fontsize=30)
    for i, (p, o) in enumerate(zip(y, odd_values)):
        if odds:
            ax.text(i, p, str(o), ha="center", va="bottom", fontsize=20)
        else:
            ax.text(i, p, str(round(p * 100)) + " %", ha="center", va="bottom", fontsize=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_diff_prob(A, home_team, away_team, ax=None, odds=False):
    """
    Plot probability of difference between home and away goals
    """
    diffs = {x: np.diag(A.T, x).sum() for x in np.arange(-4, 5)}
    x = list(diffs.keys())
    y = list(diffs.values())
    odd_values = (1 / np.array(y)).round(2)
    if ax is None:
        ax = plt.axes()
    g = sns.barplot(x=x, y=y, hue=y, palette="magma", dodge=False, ax=ax)
    g.legend_.remove()
    ax.set_yticks([])
    # ax.set_xticks(np.arange())
    ax.set_xticklabels(x, fontsize=25)
    ax.set_title(f"P of {home_team} - {away_team} goal differences", fontsize=30)
    for l in ["left", "right", "top"]:
        ax.spines[l].set_visible(False)
    for i, (p, o) in enumerate(zip(y, odd_values)):
        if odds:
            ax.text(i, p, str(o), ha="center", va="bottom", fontsize=20)
        else:
            ax.text(i, p, str(round(p * 100)) + " %", ha="center", va="bottom", fontsize=20)

        
def plot_total_score(A, cutoff=10, ax=None, odds=False):
    if ax is None:
        ax = plt.axes()
    outcome_sum = defaultdict(int)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            outcome_sum[i + j] += A[i, j]
    x = list(outcome_sum.keys())[:10]
    y = list(outcome_sum.values())[:10]
    y = np.cumsum(y)
    odd_values = (1 / y).round(2)
    sns.barplot(x=x, y=y, hue=y,palette='magma', dodge=False, ax=ax)
    ax.legend([], [], frameon=False)
    for xx, yy, o in zip(x, y, odd_values):
        if odds:
            ax.text(xx, yy, str(o), va='bottom', ha='center', fontsize=20)
        else:
            ax.text(xx, yy, str(round(yy, 2)), va='bottom', ha='center', fontsize=20)
    ax.set_yticks([])
    ax.set_xticklabels(x, fontsize=25)
    ax.set_title('P total scores', fontsize=30)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)

        
def plot_full_predictions(A, home_team, away_team, cutoff=5, odds=False, axs=None):
    """
    Do all of the above at once
    """
    return_fig = False
    if axs is None:
        fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(30, 10), constrained_layout=True)
        return_fig = True
    plot_goal_predictions(
        A[: cutoff + 1, : cutoff + 1], home_team, away_team, ax=axs[0], odds=odds
    )
    plot_outcome_prob(A, home_team, away_team, ax=axs[1], odds=odds)
    plot_diff_prob(A, home_team, away_team, ax=axs[2], odds=odds)
    plot_total_score(A, ax=axs[3], odds=odds)
    if return_fig: 
        return fig

def plot_p_and_odds(**kwargs):
    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(30, 20), constrained_layout=True)
    plot_full_predictions(**kwargs, odds=False, axs=axs[0])
    plot_full_predictions(**kwargs, odds=True, axs=axs[1])
    return fig
    
