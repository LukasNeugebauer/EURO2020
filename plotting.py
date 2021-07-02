"""
Functions to plot predictions and possibly model fit
"""


import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from itertools import product
import numpy as np


def plot_goal_predictions(A, home_team, away_team, ax=None):
    """
    Plot the posterior probability of exact outcomes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    sns.heatmap(A, cmap='magma', cbar=False, ax=ax)
    x_max, y_max = A.shape;
    for x, y in product(range(x_max), range(y_max)):
        txt = ax.text(
            y + .5, x + .5, str(round(A[x,y] * 100, 2)) + ' %',
            ha='center', va='center', color='w', size=20
        )
        txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
    ax.set_aspect('equal')
    ax.set_xlabel(away_team, fontsize=30)
    ax.set_ylabel(home_team, fontsize=30)
    ax.set_xticks(np.arange(A.shape[1]) + .5)
    ax.set_xticklabels(np.arange(A.shape[1]), fontsize=25)
    ax.set_yticks(np.arange(A.shape[0]) + .5)
    ax.set_yticklabels(np.arange(A.shape[0]), fontsize=25)
    ax.set_title('Exact predictions', fontsize=30)
    if ax is None:
        return fig


def plot_outcome_prob(A, home_team, away_team, ax=None):
    """
    Plot the winning and draw probabilities
    """
    p_home = np.tril(A, -1).sum()
    p_away = np.triu(A, 1).sum()
    p_draw = np.diag(A).sum()
    x = np.arange(3)
    y = np.array([p_home, p_draw, p_away])
    if ax is None:
        ax = plt.axes()
    sns.barplot(x, y, hue=y, palette='magma', dodge=False, ax=ax)
    ax.legend('')
    ax.set_xticks(x)
    ax.set_yticks([])
    ax.set_xticklabels([home_team, 'Draw', away_team], fontsize=25)
    ax.set_title('Outcome probabilities', fontsize=30)
    for i, p in enumerate(y):
        ax.text(i, p, str(round(p * 100)) + ' %', ha='center', va='bottom', fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_diff_prob(A, home_team, away_team, ax=None):
    """
    Plot probability of difference between home and away goals
    """
    diffs = {x: np.diag(A.T, x).sum() for x in np.arange(-4, 5)}
    x = list(diffs.keys())
    y = list(diffs.values())
    if ax is None:
        ax = plt.axes()
    g = sns.barplot(x=x, y=y, hue=y, palette='magma', dodge=False, ax=ax)
    g.legend_.remove()
    ax.set_yticks([])
    #ax.set_xticks(np.arange())
    ax.set_xticklabels(x, fontsize=25)
    ax.set_title(f'P of {home_team} - {away_team} goal differences', fontsize=30)
    for l in ['left', 'right', 'top']:
        ax.spines[l].set_visible(False)
    for i, p in enumerate(y):
        ax.text(i, p, str(round(p * 100)) + ' %', ha='center', va='bottom', fontsize=20)


def plot_full_predictions(A, home_team, away_team, cutoff=5):
    """
    Do all of the above at once
    """
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(30, 10), constrained_layout=True)
    plot_goal_predictions(A[:cutoff + 1, :cutoff + 1], home_team, away_team, ax=axs[0])
    plot_outcome_prob(A, home_team, away_team, ax=axs[1])
    plot_diff_prob(A, home_team, away_team, ax=axs[2])
    return fig
