"""
Functions for the stan modeling part
"""


import pickle
from pystan import StanModel
from os.path import exists
from countries import get_country_list, country_to_idx
import numpy as np


def compile_or_load(filename, force=False, **compile_kwargs):
    """
    Unpickle model if it has been compiled before
    compile and pickle otherwise
    """
    picfile = filename.replace(".stan", ".pic")
    if exists(picfile) and not force:
        with open(picfile, "rb") as f:
            model = pickle.load(f)
    else:
        model = StanModel(filename, **compile_kwargs)
        with open(picfile, "wb") as f:
            pickle.dump(model, f, -1)
    return model


def prepare_stan_data(df):
    """
    Extract data from dataframe and prepare for STAN
    """
    countries = get_country_list(df)
    stan_data = {}
    stan_data["n_teams"] = len(countries)
    stan_data["n_matches"] = df.shape[0]
    stan_data["home_team"] = [country_to_idx(x, countries) for x in df.home_team]
    stan_data["away_team"] = [country_to_idx(x, countries) for x in df.away_team]
    stan_data["home_goals"] = df.home_score.values.astype(int)
    stan_data["away_goals"] = df.away_score.values.astype(int)
    stan_data["is_home"] = np.logical_not(df.neutral.values).astype(int)
    # optionally add the days for the temporal decay DC-model
    if "time" in df:
        stan_data["time"] = df.time
    return stan_data
