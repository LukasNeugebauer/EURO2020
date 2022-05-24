"""
Lists and dicts for countries and functions that relate to them
"""


import numpy as np


def get_country_list(df):
    return list(np.unique(df[["home_team", "away_team"]].values.flatten()))


def country_to_idx(country, countries):
    """
    Return country_idx for a specific country. The +1 is because Stan is zero based.
    """
    if country not in countries:
        raise RuntimeError(country + " not in list")
    return countries.index(country) + 1


em_countries_by_group = {
    "A": ["Turkey", "Italy", "Wales", "Switzerland"],
    "B": ["Denmark", "Finland", "Belgium", "Russia"],
    "C": ["Netherlands", "Ukraine", "Austria", "North" "Macedonia"],
    "D": ["England", "Croatia", "Scotland", "Czech" "Republic"],
    "E": ["Spain", "Sweden", "Poland", "Slovakia"],
    "F": ["Hungary", "Portugal", "France", "Germany"],
}


em_countries = [
    "Turkey",
    "Italy",
    "Wales",
    "Switzerland",
    "Denmark",
    "Finland",
    "Belgium",
    "Russia",
    "Netherlands",
    "Ukraine",
    "Austria",
    "North Macedonia",
    "England",
    "Croatia",
    "Scotland",
    "Czech Republic",
    "Spain",
    "Sweden",
    "Poland",
    "Slovakia",
    "Hungary",
    "Portugal",
    "France",
    "Germany",
]
