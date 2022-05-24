"""
Functions for predictions
"""


import numpy as np
from scipy.stats import poisson


"""
Predictions for the Maher model which depend on sampling from predictive distribution
"""


def count_results(pred_outcomes):
    """
    quantify predictions from posterior samples
    """
    results, counts = np.unique(pred_outcomes, axis=0, return_counts=True)
    return results, counts / counts.sum()


def counts_to_matrix(results, counts):
    size = results.max() + 1
    A = np.zeros((size, size))
    for (x, y), r in zip(results, counts):
        A[x, y] = r
    return A


def predict_outcomes_maher(
    attack, defense, home_advantage, intercept, location, countries
):
    log_rate_home = intercept + attack[countries[0]] - defense[countries[1]]
    log_rate_away = intercept + attack[countries[1]] - defense[countries[0]]
    if countries[0] == location:
        log_rate_home += home_advantage
    elif countries[1] == location:
        log_rate_away += home_advantage
    rates = np.exp(np.stack([log_rate_home, log_rate_away], axis=1))
    predicted_outcomes = poisson(rates).rvs()
    results, counts = count_results(predicted_outcomes)
    return counts_to_matrix(results, counts)


"""
Predictions for the Dixon-Coles model which depends on computing posterior probability of outcomes
directly per sample
"""


def _tau(home_goals, away_goals, rho, rate_home, rate_away):
    tau = 1
    if home_goals == 0 and away_goals == 0:
        tau -= rate_home * rate_away * rho
    elif home_goals == 0 and away_goals == 1:
        tau += rate_home * rho
    elif home_goals == 1 and away_goals == 0:
        tau += rate_away * rho
    elif home_goals == 1 and away_goals == 1:
        tau -= rho
    return tau


def predict_outcomes_dixon_coles(
    attack_home,
    attack_away,
    defense_home,
    defense_away,
    intercept,
    home_advantage,
    rho,
    home_team,
    away_team,
    location,
):
    """
    Predict future outcomes from dixon coles model fit
    Not as straightforward because no clue how to sample from the DC model
    But since E(f(x)) = S(f(x) * p(x)) dx, we should just be able to compute
    the probabilities per sample and then average them
    """
    N_GOALS = 15
    # construct a grid of 15x15 which corresponds to 0-14 possible goals
    # which honestly is already complete overkill but shouldn't hurt
    home_goals, away_goals = [
        X.flatten() for X in np.meshgrid(np.arange(N_GOALS), np.arange(N_GOALS))
    ]
    log_rate_home = intercept + attack_home - defense_away
    log_rate_away = intercept + attack_away - defense_home
    if home_team == location:
        log_rate_home += home_advantage
    if away_team == location:
        log_rate_away += home_advantage
    rate_home = np.exp(log_rate_home)
    rate_away = np.exp(log_rate_away)
    poi_home = poisson(rate_home[:, None])
    poi_away = poisson(rate_away[:, None])
    A = np.zeros((N_GOALS, N_GOALS))
    for x, y in zip(home_goals, away_goals):
        tau = _tau(x, y, rho, rate_home, log_rate_away)
        p = poi_home.pmf(x) * poi_away.pmf(y) * tau
        A[x, y] = p.mean()
    return A / A.sum()


def predict_outcomes_dixon_coles_et(
    attack_home,
    attack_away,
    defense_home,
    defense_away,
    intercept,
    home_advantage,
    rho,
    home_team,
    away_team,
    location,
):
    """
    This one predicts outcomes assuming there's extra time (2 * 15) minutes
    Assuming that goals are equally distributed in the extra time and the regular time
    Just scales the poisson rates accordingly
    """
    N_GOALS = 15
    # construct a grid of 15x15 which corresponds to 0-14 possible goals
    # which honestly is already complete overkill but shouldn't hurt
    home_goals, away_goals = [
        X.flatten() for X in np.meshgrid(np.arange(N_GOALS), np.arange(N_GOALS))
    ]
    log_rate_home = intercept + attack_home - defense_away
    log_rate_away = intercept + attack_away - defense_home
    if home_team == location:
        log_rate_home += home_advantage
    if away_team == location:
        log_rate_away += home_advantage
    # extra time is 2 * 15 minutes, so we should just be able to multiply the poisson rates with 4/3
    rate_home = np.exp(log_rate_home) * 4 / 3
    rate_away = np.exp(log_rate_away) * 4 / 3
    poi_home = poisson(rate_home[:, None])
    poi_away = poisson(rate_away[:, None])
    A = np.zeros((N_GOALS, N_GOALS))
    for x, y in zip(home_goals, away_goals):
        tau = _tau(x, y, rho, rate_home, log_rate_away)
        p = poi_home.pmf(x) * poi_away.pmf(y) * tau
        A[x, y] = p.mean()
    return A / A.sum()
