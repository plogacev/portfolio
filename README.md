# Self-Contained Data Science Projects 

## Time Series Elasticity Analysis

A synthetic sales dataset is constructed with latent growth, seasonal patterns, and a random walk component to simulate external influences. Sales are modeled as a Poisson process with price elasticity effects, ensuring realistic demand shifts. A flexible Bayesian model, implemented in numpyro, is used to estimate the underlying components of the sales time series. Inference is performed using MCMC sampling. Despite structural differences, the model is flexible enough to meaningfully separate conceptually distinct components of the sales time series, such as trend, monthly and weekly seasonality, as well as price elasticity of demand. The results confirm that the model successfully reconstructs these components, demonstrating its suitability for sales forecasting.

🔗 [Full Project Details](time_series_elasticity/README.md)  

## Simple A/B Test

[Full Project Details](ab_test_1/README.md)