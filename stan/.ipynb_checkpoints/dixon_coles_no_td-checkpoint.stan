// Dixon Coles model, no temporal decay, no sum-to-0 constraint
// This one is a bit more complicated by introducing an independence correction
// The full model also has a temporal decay term which seems rather hard to implement
// Here's the paper:
// http://web.math.ku.dk/~rolf/teaching/thesis/DixonColes.pdf

functions{
  // compute the correction term
  real log_tau_correction(int x, int y, real rho, real rate_home, real rate_away){
    real tau = 1;
    if (x == 0 && y == 0){
     tau = tau - rate_home * rate_away * rho;
    }else if(x == 0 && y == 1){
      tau = tau + rate_home * rho;
    }else if(x == 1 && y == 0){
      tau = tau + rate_away * rho;
    }else if(x == 1 && y == 1){
      tau = tau - rho;
    }
    return log(tau);
  }

  //custom pmf including correction
  real dixon_coles_lpmf(int[] goals, real rho, real log_rate_home, real log_rate_away){
    int home_goals = goals[1];
    int away_goals = goals[2];
    real logp_home = poisson_log_lpmf(home_goals | log_rate_home);
    real logp_away = poisson_log_lpmf(away_goals | log_rate_away);
    real log_tau = log_tau_correction(home_goals, away_goals, rho, exp(log_rate_home), exp(log_rate_away));
    return logp_home + logp_away + log_tau;
  }
}

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
  real home_advantage; //factoring in that teams do better at home
  real intercept;
  vector[n_teams] attack;
  vector[n_teams] defense;
  //constraint on rho enforced in transformed parameters block, cmp p. 270 in the paper
  real<lower=0, upper=1> rho_raw;
}

transformed parameters{
  //enforce constraint on rho by sampling on the range [0, 1] and shifting into the boundaries
  //lower bound: max( - 1/lambda, -1/mu)
  //upper bound: min(1/lambda * mu, 1)
  real rho;
  {
    vector[n_matches] log_rate_home = (
      intercept + attack[home_team] - defense[away_team] + (home_advantage * is_home)
    );
    vector[n_matches] log_rate_away = (
      intercept + attack[away_team] - defense[home_team]
    );
    real upper_bound = min({min(1. ./ exp(log_rate_home + log_rate_away)), 1});
    real lower_bound = max({max(-1. ./ exp(log_rate_home)), max(-1. ./ exp(log_rate_away))});
    real range = upper_bound - lower_bound;
    rho = lower_bound + rho_raw * range;
  }
}

model{
  vector[n_matches] log_rate_home = (
    intercept + attack[home_team] - defense[away_team] + (home_advantage * is_home)
  );
  vector[n_matches] log_rate_away = (
    intercept + attack[away_team] - defense[home_team]
  );

  home_advantage ~ normal(0, 3);
  intercept ~ normal(0, 3);
  attack ~ std_normal();
  defense~ std_normal();
  rho_raw ~ uniform(0, 1);

  // update model with log likelihood of data
  for (m in 1:n_matches){
    int goals[2] = {home_goals[m], away_goals[m]};
    target += dixon_coles_lpmf(goals | rho, log_rate_home[m], log_rate_away[m]);
  }

}

generated quantities{
  vector[n_matches] log_lik;
  {
    vector[n_matches] log_rate_home = (
      intercept + attack[home_team] - defense[away_team] + (home_advantage * is_home)
    );
    vector[n_matches] log_rate_away = (
      intercept + attack[away_team] - defense[home_team]
    );
    // update model with log likelihood of data
    for (m in 1:n_matches){
      int goals[2] = {home_goals[m], away_goals[m]};
      log_lik[m] = dixon_coles_lpmf(goals | rho, log_rate_home[m], log_rate_away[m]);
    }
  }
}
