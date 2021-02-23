data {
  int<lower=0> n_u;  // Number of users    
  int<lower=0> n_w;  // Number of words
  int<lower=0> n_o;  // Number of observations
  int<lower=0> n;
  real x[n_w, n_u, n_o];
  int y[n_w, n_u, n_o];
}
transformed data {
  vector[n] x_arr;
  int y_arr[n];
  
  x_arr = to_vector(to_array_1d(x));
  y_arr = to_array_1d(y);
}
parameters {
  real<lower=0> sg_u;
  real<lower=0> sg_w;
  real<lower=0> sg_theta;
  real mu;

  real W[n_w];
  real U[n_u];
  real theta_wu[n_w, n_u];
}
model {
  real theta[n_w, n_u, n_o];
  vector[n] theta_arr;
  vector[n] rate;
  vector[n] exponent;
  vector[n] p;
  // vector[n] p_bn;
  
  sg_w ~ normal(0, 1);
  sg_u ~ normal(0, 1);
  sg_theta ~ normal(0, 1);
  mu ~ normal(0, 10);
  
  U ~ normal(0, sg_u);
  W ~ normal(0, sg_w);

  for (u in 1:n_u) {
    for (w in 1:n_w) {
      theta_wu[w, u] ~ normal(mu + U[u] + W[w], sg_theta);
      theta[w, u] = to_array_1d(rep_vector(theta_wu[w, u], n_o)); 
    }
  }
  
  theta_arr = to_vector(to_array_1d(theta));
  rate = exp(theta_arr);
  exponent = -rate .* x_arr;
  p = exp(exponent);
  // p_bn = pow(p, y_arr) .* pow(1-p, 1-y_arr); 
  y_arr ~ bernoulli(p);
}
