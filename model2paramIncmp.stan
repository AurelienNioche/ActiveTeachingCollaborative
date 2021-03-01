data {
  int<lower=0> n_u;    
  int<lower=0> n_w;  
  int<lower=0> n_o;  
  real delta[n_u, n_o];
  int n_rep[n_u, n_o];
  int y[n_u, n_o];
  int words[n_u, n_o];
  int n_o_by_u[n_u];
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
  int n_ou;
  real eps;
  
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
  
  eps = machine_precision();
  
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
    n_ou = n_o_by_u[u];
    words_u[1:n_ou] = words[u, 1:n_ou];
    mu1_u[1:n_ou] = rep_vector(mu1 + U1[u], n_ou) + to_vector(W1[words_u[1:n_ou]]);
    mu2_u[1:n_ou] = rep_vector(mu2 + U2[u], n_ou) + to_vector(W2[words_u[1:n_ou]]);
    theta1_wu[u, 1:n_ou] ~ normal(mu1_u[1:n_ou], sg_theta1);
    theta2_wu[u, 1:n_ou] ~ normal(mu2_u[1:n_ou], sg_theta2);
    a[1:n_ou] = exp(to_vector(theta1_wu[u, 1:n_ou]));
    b[1:n_ou] = inv_logit(to_vector(theta2_wu[u, 1:n_ou]));
    n_rep_arr[1:n_ou] = to_vector(n_rep[u, 1:n_ou]);
    delta_arr[1:n_ou] = to_vector(delta[u, 1:n_ou]);
    y_arr[1:n_ou] = to_vector(y[u, 1:n_ou]);
    exponent[1:n_ou] = - a[1:n_ou] .* pow(1-b[1:n_ou], n_rep_arr[1:n_ou]) .* delta_arr[1:n_ou];
    p_suc[1:n_ou] = exp(exponent[1:n_ou]);
    p[1:n_ou] = pow(p_suc[1:n_ou], y_arr[1:n_ou]) .* pow(1-p_suc[1:n_ou], 1-y_arr[1:n_ou]);
    log_p[1:n_ou] = log(p[1:n_ou]+eps);
    target += sum(log_p[1:n_ou]);
  }
}
