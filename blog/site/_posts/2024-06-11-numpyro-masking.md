---
layout: notebook
permalink: /2024-06-11-numpyro-masking
filename: "MaskHandlerExample.ipynb"
title: Masked Observations with NumPyro
category: "ML"
tags: ["Numpyro", "Bayesian", ]
---

We are interested in building a factorized model in NumPyro that is a highly simplified version of retail shopping, where a customer with features $X$ decides whether or not to visit a store and which items in the store to purchase if they do visit. NumPyro provides an effect handler `mask` that seems like it could handle the job. This notebook demonstrates how to use the effect handler to correctly estimate the model parameters.


## Model description
Let $X \in \mathbb{R}^D$  be a vector of features of length $D$.

We aim to model the joint probability $P(Y_0, Y_1 | X)$ where:
 - $Y_0$ is a binary outcome.
 - $Y_1 \in \{0, 1\}^I$ is a vector of binary outcomes of length $I$.
 
 
The model assumes that $Y_0 = 0$ implies all values in $Y_1$ are unobserved. This allows us to decompose the joint probability as
$$
P(Y_0, Y_1 \mid X)= P(Y_0 \mid X) \times P(Y_1 \mid Y_0 = 1, X)
$$



### Modeling $P(Y_0 | X)$

We use a logistic regression model to model $P(Y_0 \mid X)$:
$$
\text{logit}(P(Y_0 = 1 \mid X)) = X^\top \beta_{Y_0} + \alpha_{Y_0}
$$

Thus,
$$
P(Y_0 = 1 | X) = \frac{1}{1 + \exp(-(X^\top \beta_{Y_0} + \alpha_{Y_0}))},
$$
and
$$
P(Y_0 = 0 | X) = 1 - P(Y_0 = 1 | X).
$$



### Modeling $P(Y_1 | Y_0 = 1, X))$

Conditioned on $Y_0 = 1$, we model $Y_1$ as independent binary outcomes using logistic regression:
$$
\text{logit}(P(Y_{1,i} = 1 | Y_0 = 1, X)) = X^\top \beta_{Y_{1,i}} + \alpha_{Y_{1,i}}, \quad \forall i \in \{1, \ldots, n_{\text{I}}\},
$$

Thus,
$$
P(Y_{1,i} = 1 | Y_0 = 1, X) = \frac{1}{1 + \exp(-X^\top \beta_{Y_{1,i}} - \alpha_{Y_{1,i}})},
$$
and
$$
P(Y_{1,i} = 0 | Y_0 = 1, X) = 1 - P(Y_{1,i} = 1 | Y_0 = 1, X).
$$

Combining these, the joint probability becomes:
$$
P(Y_0, Y_1 | X) =
\begin{cases} 
P(Y_0 = 0 | X) & \text{if } Y_0 = 0, Y_1 = 0, \\
P(Y_0 = 1 | X) \times \prod_{i=1}^{n_{\text{products}}} P(Y_{1,i} | Y_0 = 1, X) & \text{if } Y_0 = 1.
\end{cases}
$$

# Simulated dataset
We simulate some data according to this model


```python
import numpy as np
import jax.numpy as jnp
import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import mask
import pandas as pd
from scipy import stats


# Generate some synthetic data
np.random.seed(0)
N = 10000 # number of observations
D = 3 # number of features
I = 2# number of items

X = np.random.randn(N, D)

# Generate synthetic true parameters
true_beta_Y0 = np.random.randn(1, D)
true_intercept_Y0 = 10*np.random.randn(1)
logits_Y0 = np.dot(X, true_beta_Y0.T).flatten() + true_intercept_Y0
probabilities_Y0 = 1 / (1 + np.exp(-logits_Y0))
y0 = np.random.binomial(1, probabilities_Y0)

true_beta_Y1_given_Y0 = np.random.randn(I, D)
true_intercept_Y1_given_Y0 = np.random.randn(I)
X_Y0 = X[y0 == 1]

logits_Y1_given_Y0 = np.dot(X_Y0, true_beta_Y1_given_Y0.T) + true_intercept_Y1_given_Y0
probabilities_Y1_given_Y0 = 1 / (1 + np.exp(-logits_Y1_given_Y0))
y1 = np.zeros((N, I), dtype=int)
y1[y0 == 1] = np.random.binomial(1, probabilities_Y1_given_Y0)

# Convert data to JAX arrays
X = jnp.array(X)
y0 = jnp.array(y0)
y1 = jnp.array(y1)
```

# NumPyro Models

```python
def logit_choice_model(X, name_prefix, n_outputs):
    n_features = X.shape[1]
    beta = numpyro.sample(f'{name_prefix}_beta', dist.Normal(jnp.zeros((n_outputs, n_features)), jnp.ones((n_outputs, n_features))))
    intercept = numpyro.sample(f'{name_prefix}_intercept', dist.Normal(jnp.zeros(n_outputs), 1.))
    linear_combination = jnp.einsum('ij,kj->ik', X, beta) + intercept
    return jax.nn.sigmoid(linear_combination)

def simple_model(X, I, y0=None, y1=None):
    """ This model neglects to mask any observations where Y_0=0 but it's a good baseline to get the code working.
    Parameter estimates are expected to be biased.
    """
    # Model P(Y0 | X)
    P_Y0 = logit_choice_model(X, 'Y0', 1).squeeze()

    # Sample Y0
    y0_sample = numpyro.sample('y0', dist.Bernoulli(P_Y0), obs=y0)  

    # Model P(Y1 | Y0 = 1, X)
    P_Y1_given_Y0 = logit_choice_model(X, 'Y1_given_Y0', I)  

    with numpyro.plate('products', I, dim=-1):
        with numpyro.plate('data_y1', X.shape[0]):
               numpyro.sample('y1', dist.Bernoulli(P_Y1_given_Y0), obs=y1)

                

def mask_handler_model(X, I, y0=None, y1=None):
    """This model uses the mask effect handler to mask observations where y_0=0 to estimate the correct model parameters .  
    """
    # Model P(Y0 | X)
    P_Y0 = logit_choice_model(X, 'Y0', 1).squeeze()

    # Sample Y0
    y0_sample = numpyro.sample('y0', dist.Bernoulli(P_Y0), obs=y0)  

    # Masking to filter out Y1 calculations when Y0 is 0
    mask_array = (y0_sample == 1)[:, None]

    # Model P(Y1 | Y0 = 1, X)
    P_Y1_given_Y0 = logit_choice_model(X, 'Y1_given_Y0', I)  

    with numpyro.plate('products', I, dim=-1):
        with numpyro.plate('data_y1', X.shape[0]):
            with mask(mask=mask_array):
               numpyro.sample('y1', dist.Bernoulli(P_Y1_given_Y0), obs=y1)



def get_predictive_posterior_samples(model):          
    # Define the NUTS sampler
    nuts_kernel = NUTS(model)

    # Run MCMC to sample from the posterior
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0), X, I, y0, y1)
    return mcmc
```

## Parameter Estimation

```python
simple_param_estimates = get_predictive_posterior_samples(simple_model)
mask_handler_param_estimates = get_predictive_posterior_samples(mask_handler_model)
```

    sample: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:07<00:00, 190.44it/s, 7 steps of size 6.25e-01. acc. prob=0.89]
    sample: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:08<00:00, 180.05it/s, 7 steps of size 5.28e-01. acc. prob=0.91]

```python
param_names = ["Y0_beta", "Y0_intercept", "Y1_given_Y0_beta", "Y1_given_Y0_intercept"]

ground_truth = {
 "Y0_beta": true_beta_Y0,
 "Y0_intercept": true_intercept_Y0, 
 "Y1_given_Y0_beta": true_beta_Y1_given_Y0, 
 "Y1_given_Y0_intercept": true_intercept_Y1_given_Y0 
}
    

sites = simple_param_estimates._states[simple_param_estimates._sample_field]
simple_param_summary_stats = numpyro.diagnostics.summary(sites)

sites = mask_handler_param_estimates._states[mask_handler_param_estimates._sample_field]
mask_handler_param_summary_stats = numpyro.diagnostics.summary(sites)
```

```python
for param_name in param_names:
    print(param_name)
    param_stats = pd.DataFrame({key: value.flatten() for key, value in simple_param_summary_stats[param_name].items()})
    param_stats.insert(0, 'ground_truth',ground_truth[param_name].flatten())
    print(param_stats.iloc[:,:3])
    print()
```

    Y0_beta
       ground_truth      mean       std
    0      0.371232  0.355485  0.024675
    1      0.304784  0.290453  0.025540
    2      0.504125  0.475702  0.027870
    
    Y0_intercept
       ground_truth      mean       std
    0      1.352996  1.345138  0.026762
    
    Y1_given_Y0_beta
       ground_truth      mean       std
    0     -2.474242 -1.030060  0.029048
    1     -1.462732 -0.525774  0.025773
    2      1.257190  0.879122  0.026619
    3      2.196709  1.466483  0.032559
    4     -0.646848 -0.204878  0.025808
    5      0.477828  0.511528  0.025175
    
    Y1_given_Y0_intercept
       ground_truth      mean       std
    0      0.448704 -0.461200  0.024793
    1      0.887091  0.016012  0.023572


From inspection, we can see the parameter estimates for our simple model are not correct. Now let's look at the model with masking.

```python
for param_name in param_names:
    print(param_name)
    param_stats = pd.DataFrame({key: value.flatten() for key, value in mask_handler_param_summary_stats[param_name].items()})
    param_stats.insert(0, 'ground_truth',ground_truth[param_name].flatten())
    print(param_stats.iloc[:,:3])
    print()
```

    Y0_beta
       ground_truth      mean       std
    0      0.371232  0.354625  0.025204
    1      0.304784  0.289758  0.025788
    2      0.504125  0.475834  0.025870
    
    Y0_intercept
       ground_truth      mean       std
    0      1.352996  1.345155  0.025662
    
    Y1_given_Y0_beta
       ground_truth      mean       std
    0     -2.474242 -2.559169  0.060795
    1     -1.462732 -1.450007  0.046866
    2      1.257190  1.282532  0.043331
    3      2.196709  2.290085  0.053606
    4     -0.646848 -0.649854  0.037008
    5      0.477828  0.455630  0.035820
    
    Y1_given_Y0_intercept
       ground_truth      mean       std
    0      0.448704  0.516614  0.038727
    1      0.887091  0.953112  0.034819

