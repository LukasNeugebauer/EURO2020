"""
Functions for loading and preprocessing data
"""


import pandas as pd


def get_data():
    """
    Thankfully a nice guy curated a complete list of football national team games
    We'll drop some irrelevant data, i.e. data we won't use in the model
    Also compute the home advantage
    """
    df = pd.read_csv(
        'https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
        header=0,
        index_col=0
    )
    df = df.drop(['tournament', 'city'], axis=1)
    return df.loc[(1 - df.away_score.isna()).astype(bool)]


def cut_data(df, start_idx='2016-01-01'):
    """
    For computational and model reasons, we won't keep all the data
    by default, we'll keep everything from 2016 onwards
    """
    return df.loc[start_idx:]
