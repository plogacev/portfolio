---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Generating Synthetic Order Volume

## Introduction

+++

## Import Required Libraries And Define Functions

```{code-cell} ipython3
import os
import sys

# Set XLA_FLAGS before JAX is imported
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
```

```{code-cell} ipython3
import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params
```

```{code-cell} ipython3
import polars as pl
import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_line, labs, theme_minimal, theme_bw, scale_x_continuous, scale_x_discrete, scale_x_datetime
```

```{code-cell} ipython3
import os
import polars as pl
import jax.numpy as jnp
import jax.random as random

import numpyro
```

```{code-cell} ipython3
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, AutoLaplaceApproximation
import patsy
import matplotlib.pyplot as plt
import arviz as az
from numpyro.infer import MCMC, NUTS, HMC, HMCECS, MCMC, NUTS, SA, SVI, Trace_ELBO, init_to_value
import arviz as az
```

## 1. Model

### 1.1 Auxiliary Functions

- Various functions for data transformations

```{code-cell} ipython3
def periodic_rbf(x, mu, sigma):
    """
    Computes a periodic Gaussian radial basis function (RBF).
    
    Args:
        x: Scaled day-of-year values (range [0,1]).
        mu: Center of the Gaussian basis function.
        sigma: Controls the spread of the Gaussian.
    
    Returns:
        RBF values preserving periodicity.
    """
    periodic_distance = jnp.minimum(jnp.abs(x - mu), 1 - jnp.abs(x - mu))  # Cyclic distance
    return jnp.exp(- (periodic_distance ** 2) / (2 * sigma ** 2))

def compute_doy_basis(yday_fraction, sigma = 30/365.25, n_centers = 12):
    """
    Computes 12 periodic Gaussian basis functions for seasonal effects.
    
    Args:
        yday_fraction: Normalized day of the year (range [0,1]).
        yday_factor: Scaling factor for basis function width.
    
    Returns:
        A JAX array with 12 columns representing the 12 monthly basis functions.
    """
    # Define centers of Gaussian basis functions
    month_centers = jnp.linspace( 1/(2*n_centers), 1-1/(2*n_centers), n_centers)
    
    # Generate an array of shape (length of input, 12) with the RBF values
    doy_basis = jnp.stack([periodic_rbf(yday_fraction, mu, sigma) for mu in month_centers], axis=-1)

    # Subtract each row's mean to enforce sum-to-zero constraint
    doy_basis_centered = doy_basis - jnp.mean(doy_basis, axis=-1, keepdims=True)
    
    return doy_basis_centered

def read_data(fname):
    """
    """
    df = pl.read_csv(fname)
    df = df.with_columns( pl.col("date").str.to_date())

    sales = df["sales"].to_numpy()
    log_price = df["log_price"].to_numpy()
    wday = df["date"].dt.weekday().to_numpy()
    yday = df["date"].dt.ordinal_day().to_numpy()
    is_leap_year = df["date"].dt.is_leap_year().to_numpy()
    yday_fraction = yday / (365 + is_leap_year)
    
    return {
        "sales": sales,
        "log_price": log_price,
        "wday": wday,
        "yday_fraction": yday_fraction
    }
    
def init_values(sales: jnp.array, log_price_centered: jnp.array, wday, yday_fraction: jnp.array, downsampling_factor = 1):
    """
    """
    # to-do: implement downsampling_factor
    log_state_est = jnp.log(sales)
    log_state_mean_est = jnp.mean(log_state_est)
    log_state_delta_est = jnp.diff(log_state_est)
    log_state_delta_sd_est = jnp.std(log_state_delta_est)

    return {
        "log_sigma": jnp.log( log_state_delta_sd_est ),
        "log_state_mean": log_state_mean_est,
        "log_state_delta": log_state_delta_est,
        "wday_coefficients": jnp.array([0.0]*6),
        "yday_coefficients": jnp.array([0.0]*12),
        "log_elasticity": jnp.array([0.0])
    }
```

### 1.2 Model

- The actual model

```{code-cell} ipython3
def model_local_level_poisson(sales: jnp.array, log_price_centered: jnp.array, wday, yday_fraction: jnp.array, 
                              contrasts_sdif_t: jnp.array, contrasts_wday: jnp.array, contrasts_yday: jnp.array, 
                              downsampling_factor = 1):
    """
    """

    n_obs = len(sales)
    n_states = contrasts_sdif_t.shape[0]
 
    def sample_random_walk(contrasts_sdif_t, n_states):
        log_sigma = numpyro.sample("log_sigma", dist.Gumbel(0, 1))
        sigma = numpyro.deterministic("sigma", jnp.exp(log_sigma))
        log_state_mean = numpyro.sample("log_state_mean", dist.Normal(0, 5))
        log_state_delta = numpyro.sample( "log_state_delta", dist.Normal(0, 1), sample_shape=(n_states-1,))
        log_state_base = numpyro.deterministic("log_state_base", jnp.dot(contrasts_sdif_t, log_state_delta) * sigma + log_state_mean )
        return log_state_base

    def sample_downsampled_random_walk(contrasts_sdif_t, n_obs, n_states):
        log_state_base_downsampled = sample_random_walk(contrasts_sdif_t, n_states)
        
        idx_n_weight = jnp.array(range(0, n_obs))/downsampling_factor
        idx_1 = jnp.array( jnp.floor(idx_n_weight), dtype=int)
        idx_2 = jnp.array( jnp.ceil(idx_n_weight), dtype=int)
        weight_2 = idx_n_weight - idx_1

        state_before = log_state_base_downsampled[idx_1]
        state_after = log_state_base_downsampled[idx_2]
        return (1-weight_2)*state_before + weight_2*state_after
        
    def sample_wday_effect(contrasts_wday, wday):
        # Prior for day-of-the-week effects (6 coefficients)
        wday_coefficients = numpyro.sample("wday_coefficients", dist.Normal(0, 1), sample_shape=(6,))

        # Compute wday effect per observation (sum-to-zero constraint applied via contrasts)
        wday_effects = jnp.dot(contrasts_wday, wday_coefficients)
        return jnp.array([wday_effects[d - 1] for d in wday]) # to-do: just use an index vector instead of a loop

    def sample_yday_effect(contrasts_yday, yday_fraction):
        # Prior for yearly seasonality effects (12 coefficients)
        yday_coefficients = numpyro.sample("yday_coefficients", dist.Normal(0, 1), sample_shape=(12,))
        return jnp.dot(contrasts_yday, yday_coefficients)

    def sample_price_effect(log_price_centered):
        # Prior for price elasticity
        log_elasticity = numpyro.sample( "log_elasticity", dist.Normal(0, 1) )
        elasticity = numpyro.deterministic( "elasticity", -1 * jnp.exp( log_elasticity ))
        return log_price_centered * elasticity


    # Sample random walk    
    if n_obs == n_states:
        log_state_base = sample_random_walk(contrasts_sdif_t, n_states)
    else:
        log_state_base = sample_downsampled_random_walk(contrasts_sdif_t, n_obs, n_states)

    # Sample day-of-the-week effects
    wday_effect = sample_wday_effect(contrasts_wday, wday)

    # Sample day-of-the-year effects
    yday_effect = sample_yday_effect(contrasts_yday, yday_fraction)

    # Sample elasticity effect
    price_effect = sample_price_effect(log_price_centered)

    # Compute state
    state = numpyro.deterministic("state", jnp.exp(log_state_base + price_effect + yday_effect + wday_effect))

    # Compute log-likelihood for poisson emissions
    numpyro.sample("sales", dist.Poisson(rate=state), obs=sales)
```

### 1.3 Model Fitting Logic

- Functions to fit the model

```{code-cell} ipython3
def prepare_model_arguments(sales: jnp.array, log_price: jnp.array, wday, yday_fraction: jnp.array, downsampling_factor = 1):
    """ 
    """    
    n_obs = len(sales)
    if downsampling_factor == 1:
        n_states = n_obs
    else:
        n_states = int( np.floor(n_obs/downsampling_factor) + 1 ) 
    
    # Define contrast matrix for random walk (T coefficients, sum-to-zero constraint)
    contrasts_sdif_t = patsy.contrasts.Diff().code_without_intercept(range(0, n_states)).matrix

    # Define contrast matrix for day-of-the-week effects (6 coefficients, sum-to-zero constraint)
    contrasts_wday = patsy.contrasts.Diff().code_without_intercept(range(0,7)).matrix  # 7 days → 6 contrasts

    # Compute yday effect per observation (sum-to-zero constraint applied via contrasts)
    contrasts_yday = compute_doy_basis(yday_fraction, sigma = 30/365.25, n_centers = 12)

    # Compute centered log price differences
    log_price_centered = log_price - jnp.mean(log_price)

    # Set up the model parameters
    model_arguments = {'sales': sales, 'log_price_centered': log_price_centered, 'wday': wday, 'yday_fraction': yday_fraction,
                       'downsampling_factor': downsampling_factor,
                       'contrasts_sdif_t': contrasts_sdif_t, 'contrasts_wday': contrasts_wday, 'contrasts_yday': contrasts_yday}
    
    # Prepare init values for parameters 
    init_params = init_values(sales, log_price_centered, wday, yday_fraction)

    return init_params, model_arguments
```

```{code-cell} ipython3
def run_svi(sales: jnp.array, log_price: jnp.array, wday, yday_fraction: jnp.array, num_samples=1_000, num_steps=10_000):
        """ """
        rng_key = random.PRNGKey(seed=42)

        n_obs = len(sales)
        
        # Prepare model arguments
        init_params, model_arguments = prepare_model_arguments(sales, log_price, wday, yday_fraction)    

        model = model_local_level_poisson
        guide = AutoNormal(model=model) # AutoLaplaceApproximation(model=model) # AutoNormal(model=model) # guide = AutoMultivariateNormal(model=model)
        optimizer = numpyro.optim.Adam(step_size=0.01)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        rng_key, rng_subkey = random.split(key=rng_key)

        svi_result = svi.run(rng_key = rng_subkey, num_steps = num_steps, **model_arguments, init_params = init_params)
        params = svi_result.params

        # get posterior samples (parameters)
        rng_key, rng_subkey = random.split(key=rng_key)
        predictive = Predictive(model=guide, params=params, num_samples=num_samples)
        posterior_parameters = predictive(rng_subkey, **model_arguments)

        # get posterior predictive (deterministics and likelihood)
        rng_key, rng_subkey = random.split(key=rng_key)
        predictive = Predictive(model=model, guide=guide, params=params, num_samples=num_samples)
        posterior_generated = predictive(rng_subkey, **model_arguments)

        return svi_result, posterior_parameters, posterior_generated
```

```{code-cell} ipython3
def run_nuts(sales: jnp.array, log_price: jnp.array, wday, yday_fraction: jnp.array, downsampling_factor = 1, n_chains = 1, num_warmup=1_000, num_samples=1_000):
    """ Runs NUTS MCMC inference on the model 
    """
    rng_key = random.PRNGKey(0)
    
    n_obs = len(sales)
    
    # Prepare model arguments
    init_params, model_arguments = prepare_model_arguments(sales = sales, log_price = log_price, wday = wday, yday_fraction = yday_fraction, downsampling_factor = downsampling_factor)

    rng_key, rng_key_ = random.split(rng_key)

    numpyro.set_host_device_count(n_chains)

    reparam_model = model_local_level_poisson
    kernel = NUTS(reparam_model, step_size=0.01, max_tree_depth=8)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=n_chains)
    mcmc.run( rng_key_, **model_arguments #, init_params = init_params
             )

    return mcmc
    #samples = mcmc.get_samples()
    #
    #return {'log_state_base': jnp.mean(samples['log_state_base'], axis=0),
    #        'state': jnp.mean(samples['state'], axis=0),
    #        'mcmc': mcmc
    #       }  
  
```

## 2. Fit the Model 

```{code-cell} ipython3

data = read_data("sales_synthetic.csv")
```

```{code-cell} ipython3
m_fit = run_nuts(data['sales'], data['log_price'], data['wday'], data['yday_fraction'], downsampling_factor = 7, n_chains = 4, num_warmup=1_000, num_samples=1_000)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
summary = az.summary(m_fit)
```

```{code-cell} ipython3
#import arviz as az
#summary = az.summary(m['mcmc'])
summary.loc[['sigma', 'elasticity']]
```

```{code-cell} ipython3
summary
```

```{code-cell} ipython3
az.summary(data=idata_parameters, var_names=["log_sigma", "log_price_coefficient", "yday_coefficients", "wday_coefficients"], round_to=3)
```

```{code-cell} ipython3
import inspect
print(inspect.getsource(m['mcmc'].print_summary))
```

```{code-cell} ipython3
import jax
from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
```

```{code-cell} ipython3
import jax
from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
```

```{code-cell} ipython3
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

import blackjax
import blackjax.smc.resampling as resampling
from blackjax.smc import extend_params

from numpyro.infer.util import initialize_model
```

```{code-cell} ipython3
init_params, model_arguments = prepare_model_arguments(sales, log_price, wday, yday_fraction)    

rng_key, init_key = jax.random.split(rng_key)
init_params, potential_fn_gen, *_ = initialize_model(
    init_key,
    model_local_level_poisson,
    model_kwargs=model_arguments,
    dynamic_args=True,
)
```

```{code-cell} ipython3
logdensity_fn = lambda position: -potential_fn_gen(**model_arguments)(position)
initial_position = init_params.z
```

```{code-cell} ipython3
import blackjax

num_warmup = 2000

adapt = blackjax.window_adaptation(
    blackjax.nuts, logdensity_fn, target_acceptance_rate=0.8
)
rng_key, warmup_key = jax.random.split(rng_key)
(last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_warmup)
kernel = blackjax.nuts(logdensity_fn, **parameters).step
```

```{code-cell} ipython3
def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, (
        infos.acceptance_rate,
        infos.is_divergent,
        infos.num_integration_steps,
    )
```

```{code-cell} ipython3
num_sample = 1000
rng_key, sample_key = jax.random.split(rng_key)
states, infos = inference_loop(sample_key, kernel, last_state, num_sample)
#_ = states.position["mu"].block_until_ready()
```

```{code-cell} ipython3
states
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
svi_result, posterior_parameters, posterior_generated = run_svi(sales, log_price, wday, yday_fraction)
```

```{code-cell} ipython3
idata_parameters = az.from_dict(
     posterior = { k: np.expand_dims(a=np.asarray(v), axis=0) for k, v in posterior_parameters.items() },
 )
idata_generated = az.from_dict(
     posterior={ k: np.expand_dims(a=np.asarray(v), axis=0) for k, v in posterior_generated.items() },
 )
```

```{code-cell} ipython3
#az.summary(data=idata_generated, var_names=["sigma"], round_to=3)
az.summary(data=idata_parameters, var_names=["log_sigma", "log_price_coefficient", "yday_coefficients", "wday_coefficients"], round_to=3)
```

```{code-cell} ipython3
az.summary(data=idata_generated, var_names=["state"], round_to=3)
```

```{code-cell} ipython3
state = jnp.mean(posterior_generated['state'], axis=0).tolist()
df = df.with_columns([ pl.Series("state", state) ])
df
```

```{code-cell} ipython3

```

```{code-cell} ipython3
posterior_generated
```

```{code-cell} ipython3

```

```{code-cell} ipython3
x = pd.DataFrame({ 'date': df["date"].to_numpy(), 'sales': df["sales"].to_numpy(), 'state': m['state'] })
ggplot(x, aes(x='date', y='sales')) + geom_point() + geom_line(aes(y='state'), color = "red") + theme_bw()
```
