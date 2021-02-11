functions {
  vector[] make_stuff(vector mu, matrix L, vector b, vector s, vector u) {
    int K = rows(mu); vector[K] d; vector[K] z; vector[K] out[2];
    for (k in 1:K) {
      int km1 = k - 1;
      if (s[k] != 0) {
        real z_star = (b[k] - 
                      (mu[k] + ((k > 1) ? L[k,1:km1] * head(z, km1) : 0))) / 
                      L[k,k];
        real v; real u_star = Phi(z_star);
        if (s[k] == -1) {
          v = u_star * u[k];
          d[k] = u_star;
        }
        else {
          d[k] = 1 - u_star;
          v = u_star + d[k] * u[k];
        }
        z[k] = inv_Phi(v);
      }
      else {
        z[k] = inv_Phi(u[k]);
        d[k] = 1;
      }
    }
    out[1] = z;
    out[2] = d;
    return out;
  }
}
data {
  int<lower=2> K;             // number of dimensions
  vector[K] b;                // lower or upper bound
  
  // s[k] ==  0 implies no constraint; otherwise 
  // s[k] == -1 -> b[k] is an upper bound 
  // s[k] == +1 -> b[k] is a lower bound
  vector<lower=-1,upper=1>[K] s;
  
  vector[K] mu;
  cholesky_factor_cov[K,K] L;
}
parameters {
  vector<lower=0,upper=1>[K] u;
}
model {
  target += log(make_stuff(mu, L, b, s, u)[2]); // Jacobian adjustments
  // implicit: u ~ uniform(0,1)
}
generated quantities {
  vector[K] y = mu + L * make_stuff(mu, L, b, s, u)[1];
}
