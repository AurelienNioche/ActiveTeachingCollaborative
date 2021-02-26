data {
  int<lower=0> n_u;    
  int<lower=0> n_w;  
  int<lower=0> n_o;  
  real delta[n_u, n_o];
  int n_rep[n_u, n_o];
  int y[n_u, n_o];
  int words[n_u, n_o];
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
  real theta1_wu[n_u, n_o];
  real U2[n_u];
  real W2[n_w];
  real theta2_wu[n_u, n_o];
}
model {
  vector[n_o] a;
  vector[n_o] b;
  vector[n_o] rate;
  vector[n_o] exponent;
  vector[n_o] p;
  vector[n_o] p_suc;
  vector[n_o] log_p;
  vector[n_o] y_arr;
  vector[n_o] delta_arr;
  vector[n_o] n_rep_arr;
  vector[n_o] mu1_u;
  vector[n_o] mu2_u;
  int words_u[n_o];
  
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
    words_u = words[u];
    mu1_u = rep_vector(mu1 + U1[u], n_o) + to_vector(W1[words_u]);
    mu2_u = rep_vector(mu2 + U2[u], n_o) + to_vector(W2[words_u]);
    theta1_wu[u] ~ normal(mu1_u, sg_theta1);
    theta2_wu[u] ~ normal(mu2_u, sg_theta2);
    a = exp(to_vector(theta1_wu[u]));
    b = inv_logit(to_vector(theta2_wu[u]));
    n_rep_arr = to_vector(n_rep[u]);
    delta_arr = to_vector(delta[u]);
    y_arr = to_vector(y[u]);
    exponent = - a .* pow(1-b, n_rep_arr) .* delta_arr;
    p_suc = exp(exponent);
    p = pow(p_suc, y_arr) .* pow(1-p_suc, 1-y_arr);
    log_p = log(p);
    target += sum(log_p);
  }
}
