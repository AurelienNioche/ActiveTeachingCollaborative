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
  vector[n] n_rep_arr;
  vector[n] delta_arr;
  int y_arr[n];
  
  n_rep_arr = to_vector(to_array_1d(n_rep));
  delta_arr = to_vector(to_array_1d(delta));
  y_arr = to_array_1d(y);
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
  real theta1[n_w, n_u, n_o];
  vector[n] theta1_arr;
  real theta2[n_w, n_u, n_o];
  vector[n] theta2_arr;
  vector[n] a;
  vector[n] b;
  vector[n] rate;
  vector[n] exponent;
  vector[n] p;
  
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
      theta1[w, u] = to_array_1d(rep_vector(theta1_wu[w, u], n_o)); 
      theta2_wu[w, u] ~ normal(mu2 + U2[u] + W2[w], sg_theta2);
      theta2[w, u] = to_array_1d(rep_vector(theta2_wu[w, u], n_o)); 
    }
  }
  
  theta1_arr = to_vector(to_array_1d(theta1));
  theta2_arr = to_vector(to_array_1d(theta2));
  a = exp(theta1_arr);
  b = inv_logit(theta2_arr);
  exponent = - a .* pow(1-b, n_rep_arr) .* delta_arr;
  p = exp(exponent);
  y_arr ~ bernoulli(p);
}
