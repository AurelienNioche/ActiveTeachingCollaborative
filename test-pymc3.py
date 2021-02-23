import pymc3 as pm

with pm.Model() as model:
    pm.Bernoulli()

import stan.common
