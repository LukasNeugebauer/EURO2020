// Similar to maher model, but basically a logistic regression
// No home advantage here

data{
  int n_teams;
  int n_matches;
  int home_team[n_matches];
  int away_team[n_matches];
  int home_win[n_matches];
}

parameters{
  vector[n_teams - 1] attack_free;
  vector[n_teams - 1] defense_free;
}

transformed parameters{
  vector[n_teams] attack;
  vector[n_teams] defense;

  attack[1:(n_teams - 1)] = attack_free;
  defense[1:(n_teams - 1)] = defense_free;
  attack[n_teams] = -sum(attack_free);
  defense[n_teams] = -sum(defense_free);

}

model{
  vector[n_matches] home_strength = attack[home_team] - defense[away_team];
  vector[n_matches] away_strength = attack[away_team] - defense[home_team];

  attack_free ~ normal(0, 2);
  defense_free ~ normal(0, 3);

  target += bernoulli_lpmf(home_win | inv_logit(1 + home_strength - away_strength));
}
