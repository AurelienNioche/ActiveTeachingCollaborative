import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()  # Nice plot aesthetic
np.random.seed(101)


def example():
    model = """
    data {
        int<lower=0> N;
        vector[N] x;
        vector[N] y;
    }
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    model {
        y ~ normal(alpha + beta * x, sigma);
    }
    """
    
    # Parameters to be inferred
    alpha = 4.0
    beta = 0.5
    sigma = 1.0
    
    # Generate and plot data
    x = 10 * np.random.rand(100)
    y = alpha + beta * x
    y = np.random.normal(y, scale=sigma)
    
    # Put our data in a dictionary
    data = {'N': len(x), 'x': x, 'y': y}
    
    # Compile the model
    sm = pystan.StanModel(model_code=model)
    
    # Train the model and generate samples
    fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)
    
    summary_dict = fit.summary()
    df = pd.DataFrame(summary_dict['summary'], 
                      columns=summary_dict['summary_colnames'], 
                      index=summary_dict['summary_rownames'])
    
    alpha_mean, beta_mean = df['mean']['alpha'], df['mean']['beta']
    
    # Extracting traces
    alpha = fit['alpha']
    beta = fit['beta']
    sigma = fit['sigma']
    lp = fit['lp__']





def plot_trace(param, param_name='parameter'):
  """Plot the trace and posterior of a parameter."""
  
  # Summary statistics
  mean = np.mean(param)
  median = np.median(param)
  cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
  
  # Plotting
  plt.subplot(2,1,1)
  plt.plot(param)
  plt.xlabel('samples')
  plt.ylabel(param_name)
  plt.axhline(mean, color='r', lw=2, linestyle='--')
  plt.axhline(median, color='c', lw=2, linestyle='--')
  plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
  plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
  plt.title('Trace and Posterior Distribution for {}'.format(param_name))

  plt.subplot(2,1,2)
  plt.hist(param, 30, density=True); sns.kdeplot(param, shade=True)
  plt.xlabel(param_name)
  plt.ylabel('density')
  plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
  plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
  plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
  plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
  
  plt.gcf().tight_layout()
  plt.legend()
  
  
###############################################################################
  
## Sum of level + difficulty


model = """
data {
    int<lower=0> U;  // Number of users    
    int<lower=0> A;  // Number of words
    vector[2] theta[A,U];
}
parameters {
    vector[2] mu_u;
    cov_matrix[2] Sigma_u;
    cov_matrix[2] Sigma_a;
    
    cov_matrix[2] Sigma_theta;
    
    vector[2] D[A];
    vector[2] L[U];
}
model {
    L[1:U] ~ multi_normal(mu_u, Sigma_u);
    D[1:A] ~ multi_normal(rep_vector(0,2), Sigma_a);
    
    for (u in 1:U) {
        for (a in 1:A) {
            theta[a,u] ~ multi_normal(L[u] + D[a], Sigma_theta);
        }
    }
}
"""

mu_u = np.array([1.0, 2.0])
Sigma_u = .1 * np.eye(2)

mu_a = np.array([3.0, 1.0])
Sigma_a = .2 * np.eye(2)

Sigma_theta = .07 * np.eye(2)

A = 10
U = 15

L = np.random.multivariate_normal(mu_u, Sigma_u, U)
D = np.random.multivariate_normal(mu_a, Sigma_a, A)

theta = np.random.multivariate_normal(np.zeros(2), Sigma_theta, (A,U))

for a in range(A):
    for u in range(U):
        theta[a,u] = theta[a,u] + L[u] + D[a]
        
data = {'U': U, 'A': A, 'theta': theta}

# Compile the model
sm = pystan.StanModel(model_code=model)

# Train the model and generate samples
fit = sm.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)


###############################################################################
  
## Memory model



model = """
data {
    int<lower=1> U;  // Number of users    
    int<lower=1> A;  // Number of words
    int<lower=1> T; // Number of interactions
    int<lower=0, upper=U> user[T];
    int<lower=0, upper=A> word[T];
    vector[2] states[T];
    int<lower=0, upper=1> y[T];
}
parameters {
    vector[2] theta[A,U];
}
transformed parameters {
    real p[T]; 
    for (t in 1:T) 
        p[t] = exp(- theta[word[t],user[t]][0] * pow(1 - theta[word[t],user[t]][1], states[t][0]) * states[t][1]);
}
model {
    y ~ bernoulli(p);
}
"""
sm = pystan.StanModel(model_code=model)



mu_u = np.array([1.0, .5])
Sigma_u = .1 * np.eye(2)

mu_a = np.array([0, 0])
Sigma_a = .02 * np.eye(2)

Sigma_theta = .04 * np.eye(2)

A = 10
U = 15

L = np.random.multivariate_normal(mu_u, Sigma_u, U)
D = np.random.multivariate_normal(mu_a, Sigma_a, A)

theta = np.random.multivariate_normal(np.zeros(2), Sigma_theta, (A,U))

for a in range(A):
    for u in range(U):
        theta[a,u] = theta[a,u] + L[u] + D[a]
        

T = 150
word = np.random.randint(0, A, T)
user = np.random.randint(0, U, T)
last = np.random.randint(0, 10, T)
n_seen = np.random.randint(1, 5, T)
states = np.vstack((last, n_seen)).transpose()
beta = np.array([max(min(0, theta[word[t],user[t]][1]),1) for t in range(T)])
p = np.array([np.exp(- theta[word[t],user[t]][0] * np.power(1 - beta[t], states[t][0]) * states[t][1]) for t in range(T)])
y = np.random.binomial(n=1, p=p)




data = {'U': U, 'A': A, 'T':T, 'user':user, 'word':word, 'states':states, 'y':y, 'theta': theta}

# Compile the model
sm = pystan.StanModel(model_code=model)

# Train the model and generate samples
fit = sm.sampling(data=data, iter=100, chains=4, warmup=50, thin=1, seed=101)


