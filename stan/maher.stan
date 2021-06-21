// Maher model, no hierarchies, sum to 0 constraint
// The simplest model to get started. Assumes fixed attack and defense scores per team.
// Goals for team follow a poisson distribution where the parameter is based on the
// difference between own attack and opponents defense strength

data{
  int n_teams;
  int n_matches;
  int home_team[n_matches];
  int away_team[n_matches];
  int home_goals[n_matches];
  int away_goals[n_matches];
  // indicate whether the game was on neutral ground
  // it should really be int[] of 1 and 0, but stan doesn't allow real * int[]
  vector[n_matches] is_home;

}

parameters{
  real home_advantage;
  real intercept;
  vector[n_teams - 1] attack_free;
  vector[n_teams - 1] defense_free;
}

transformed parameters{
  //impose sum to 0 constraint
  vector[n_teams] attack;
  vector[n_teams] defense;
  attack[1:(n_teams - 1)] = attack_free;
  defense[1:(n_teams - 1)] = defense_free;
  attack[n_teams] = -sum(attack_free);
  defense[n_teams] = -sum(defense_free);
}

model{
  vector[n_matches] home_log_rate = (
    intercept + attack[home_team] - defense[away_team] + (home_advantage * is_home)
  );
  vector[n_matches] away_log_rate = (
    intercept + attack[away_team] - defense[home_team]
  );

  intercept ~ normal(0, 2);
  home_advantage ~ normal(0, 2);
  attack_free ~ normal(0, 3);
  defense_free ~ normal(0, 3);

  // update model with log likelihood of data
  target += poisson_log_lpmf(home_goals | home_log_rate);
  target += poisson_log_lpmf(away_goals | away_log_rate);
}

generated quantities{
  real log_lik[n_matches];
  for (match in 1:n_matches){
    real home_log_rate;
    real away_log_rate;
    home_log_rate = (
      intercept + (home_advantage * is_home[match]) + attack[home_team[match]] - defense[away_team[match]]
    );
    away_log_rate = (
      intercept + attack[away_team[match]] - defense[home_team[match]]
    );
    //compute log likelihood of results
    log_lik[match] = poisson_log_lpmf(home_goals[match] | home_log_rate) + poisson_log_lpmf(away_goals[match] | away_log_rate);
  }
}
