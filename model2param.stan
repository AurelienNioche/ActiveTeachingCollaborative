data {
  int<lower=0> n_u;  // Number of users    
  int<lower=0> n_w;  // Number of words
  int<lower=0> n_o;  // Number of observations
  int<lower=0> n;
  real delta[n_w, n_u, n_o];
  int n_rep[n_w, n_u, n_o];
  int y[n_w, n_u, n_o];
}
transformed data {
}
parameters {
  real<lower=0> sg_u1;
  real<lower=0> sg_w1;
  real<lower=0> sg_theta1;
  real mu1;
  real<lower=0> sg_u2;
  real<lower=0> sg_w2;
  real<lower=0> sg_theta2;
  real mu2;
  
  real U1[n_u];
  real W1[n_w];
  real theta1_wu[n_w, n_u];
  real U2[n_u];
  real W2[n_w];
  real theta2_wu[n_w, n_u];
}
model {
  real a;
  real b;
  vector[n_o] rate;
  vector[n_o] exponent;
  vector[n_o] p;
  vector[n_o] p_suc;
  vector[n_o] log_p;
  vector[n_o] y_arr;
  vector[n_o] delta_arr;
  vector[n_o] n_rep_arr;
  
  sg_w1 ~ inv_gamma(1, 1);
  sg_u1 ~ inv_gamma(1, 1);
  sg_theta1 ~ inv_gamma(1, 1);
  mu1 ~ normal(0, 10);
  U1 ~ normal(0, sg_u1);
  W1 ~ normal(0, sg_w1);
  
  sg_w2 ~ inv_gamma(1, 1);
  sg_u2 ~ inv_gamma(1, 1);
  sg_theta2 ~ inv_gamma(1, 1);
  mu2 ~ normal(0, 10);
  U2 ~ normal(0, sg_u2);
  W2 ~ normal(0, sg_w2);

  for (u in 1:n_u) {
    for (w in 1:n_w) {
      theta1_wu[w, u] ~ normal(mu1 + U1[u] + W1[w], sg_theta1);
      theta2_wu[w, u] ~ normal(mu2 + U2[u] + W2[w], sg_theta2);
      a = exp(theta1_wu[w, u]);
      b = inv_logit(theta2_wu[w, u]);
      n_rep_arr = to_vector(n_rep[w, u]);
      delta_arr = to_vector(delta[w, u]);
      y_arr = to_vector(y[w, u]);
      exponent = - rep_vector(a, n_o) .* pow(1-rep_vector(b, n_o), n_rep_arr) .* delta_arr;
      p_suc = exp(exponent);
      p = pow(p_suc, y_arr) .* pow(1-p_suc, 1-y_arr);
      log_p = log(p);
      target += sum(log_p);
    }
  }
}
