data {
  int<lower=0> n_u;  // Number of users    
  int<lower=0> n_w;  // Number of words
  int<lower=0> n_o;  // Number of observations
  real theta[n_w, n_u, n_o];
}
parameters {
  real<lower=0> sg_u;
  real<lower=0> sg_w;
  real<lower=0> sg_theta;
  real mu;

  real W[n_w];
  real U[n_u];
}
model {
  sg_w ~ normal(0, 1);
  sg_u ~ normal(0, 1);
  sg_theta ~ normal(0, 1);
  mu ~ normal(0, 1);
  
  U[1:n_u] ~ normal(0, sg_u);
  W[1:n_w] ~ normal(0, sg_w);

  for (u in 1:n_u) {
      for (w in 1:n_w) {
          theta[w,u] ~ normal(mu + U[u] + W[w], sg_theta);
      }
  }
}
