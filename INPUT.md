# ActiveTeachingCollaborative


## Model

We assume the following model:

\begin{align}
Z_u^\rho &\sim \mathcal{N}(0, \sigma_u^\rho)\\
Z_w^\rho &\sim \mathcal{N}(0, \sigma_w^\rho)\\
Z_{u, w}^\rho &= \mu^\rho + Z_u^\rho + Z_w^\rho \\
\end{align}
where $Z_u^{\rho}$ is a random variable whose distribution is specific to user $u$ and parameter $\rho$, and $\rho \in {\alpha, \beta}$.

The probability of recall for user $u$ and item/word $w$ at time $t$ is defined as:
\begin{align}
p(\omega = 1 \mid t, u, w) &= e^{-Z_{u, w}^\alpha (1-Z_{u, w}^\beta)^n \delta_{u, w}^t}  \\
\end{align}
where $\delta_{u, w}^t$ is the time elapsed since the last presentation for user $u$, item $w$ at time $t$.