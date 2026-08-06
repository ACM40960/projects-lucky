"""Sea ice and CO2 analysis converted from the Jupyter notebook."""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


plt.style.use("seaborn-v0_8-whitegrid")
rng = np.random.default_rng(42)


# Load data
co2_path = "co2_annmean_mlo.csv"
ice_path = "N_09_extent_v4.0.csv"

co2_raw = pd.read_csv(co2_path, comment="#")
ice_raw = pd.read_csv(ice_path)

print("CO2 data shape:", co2_raw.shape)
print(co2_raw.head())

print("Sea ice data shape:", ice_raw.shape)
print(ice_raw.head())


# Data preprocessing
def clean_column_names(df):
    '''
    Return a copy of a DataFrame with beginner-friendly, consistent column names.

    Example:
    " source_dataset" becomes "source_dataset"
    "CO2 Mean" becomes "co2_mean"
    '''
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return cleaned


co2 = clean_column_names(co2_raw)
ice = clean_column_names(ice_raw)

print("Cleaned CO2 columns:", list(co2.columns))
print("Cleaned sea ice columns:", list(ice.columns))

co2 = co2.rename(columns={"mean": "co2_ppm"})
ice = ice.rename(columns={"extent": "ice_extent_million_sq_km"})

co2_clean = co2[["year", "co2_ppm"]].copy()
ice_clean = ice[["year", "ice_extent_million_sq_km"]].copy()

for df in [co2_clean, ice_clean]:
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

co2_clean = co2_clean.dropna(subset=["year", "co2_ppm"])
ice_clean = ice_clean.dropna(subset=["year", "ice_extent_million_sq_km"])

co2_clean["year"] = co2_clean["year"].astype(int)
ice_clean["year"] = ice_clean["year"].astype(int)

print(co2_clean.head())
print(ice_clean.head())

data = pd.merge(co2_clean, ice_clean, on="year", how="inner")

print("Merged data shape:", data.shape)
print("Year range:", data["year"].min(), "to", data["year"].max())
print(data.head())
print(data.tail())


# Correlation analysis
pearson_corr = data["co2_ppm"].corr(data["ice_extent_million_sq_km"], method="pearson")
print(f"Pearson correlation between CO2 and sea ice extent: {pearson_corr:.4f}")


# Plot CO2 against sea ice
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    data["co2_ppm"],
    data["ice_extent_million_sq_km"],
    color="steelblue",
    alpha=0.85,
)
ax.set_xlabel("CO2 concentration (ppm)")
ax.set_ylabel("September sea ice extent (million sq km)")
ax.set_title("CO2 vs September Northern Hemisphere Sea Ice Extent")
plt.show()


# Fit OLS model
y = data["ice_extent_million_sq_km"]
X = sm.add_constant(data["co2_ppm"])  # add beta0 intercept column

ols_model = sm.OLS(y, X).fit()

print(ols_model.summary())


# Plot OLS fitted line
x_grid = np.linspace(data["co2_ppm"].min(), data["co2_ppm"].max(), 200)
X_grid = sm.add_constant(x_grid)
y_fitted = ols_model.predict(X_grid)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    data["co2_ppm"],
    data["ice_extent_million_sq_km"],
    color="steelblue",
    alpha=0.85,
    label="Observed years",
)
ax.plot(x_grid, y_fitted, color="crimson", linewidth=2.5, label="OLS fitted line")
ax.set_xlabel("CO2 concentration (ppm)")
ax.set_ylabel("September sea ice extent (million sq km)")
ax.set_title("OLS Regression Fit")
ax.legend()
plt.show()


# Analytical Bayesian linear regression
co2_mean = data["co2_ppm"].mean()
co2_std = data["co2_ppm"].std(ddof=0)

data["co2_z"] = (data["co2_ppm"] - co2_mean) / co2_std

X_bayes = np.column_stack([
    np.ones(len(data)),
    data["co2_z"].to_numpy()
])
y_bayes = data["ice_extent_million_sq_km"].to_numpy()

n, p = X_bayes.shape
print(f"Number of observations: {n}")
print(f"Number of regression coefficients: {p}")

beta_prior_mean = np.zeros(p)
V0 = 1000 * np.eye(p)
V0_inv = np.linalg.inv(V0)

a0 = 0.01
b0 = 0.01

Vn = np.linalg.inv(V0_inv + X_bayes.T @ X_bayes)
beta_posterior_mean = Vn @ (V0_inv @ beta_prior_mean + X_bayes.T @ y_bayes)

an = a0 + n / 2
bn = b0 + 0.5 * (
    y_bayes.T @ y_bayes
    + beta_prior_mean.T @ V0_inv @ beta_prior_mean
    - beta_posterior_mean.T @ np.linalg.inv(Vn) @ beta_posterior_mean
)

beta_posterior_cov = (bn / (an - 1)) * Vn

print("Posterior mean of beta [intercept, slope_on_standardized_CO2]:")
print(beta_posterior_mean)

print("\nPosterior covariance of beta:")
print(beta_posterior_cov)

print(f"\nPosterior sigma^2 parameters: a_n={an:.4f}, b_n={bn:.4f}")


# Draw posterior samples
n_samples = 20_000

sigma2_samples = bn / rng.gamma(shape=an, scale=1.0, size=n_samples)

beta_samples = np.empty((n_samples, p))
for i, sigma2 in enumerate(sigma2_samples):
    beta_samples[i, :] = rng.multivariate_normal(
        mean=beta_posterior_mean,
        cov=sigma2 * Vn,
    )

posterior_samples = pd.DataFrame({
    "beta0_intercept": beta_samples[:, 0],
    "beta1_slope_standardized_co2": beta_samples[:, 1],
    "sigma": np.sqrt(sigma2_samples),
})

print(posterior_samples.head())

credible_intervals = posterior_samples.quantile([0.025, 0.5, 0.975]).T
credible_intervals.columns = ["2.5%", "median", "97.5%"]

print(credible_intervals)


# Plot posterior densities
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, column, title in zip(
    axes,
    posterior_samples.columns,
    ["Posterior density: beta0", "Posterior density: beta1", "Posterior density: sigma"],
):
    ax.hist(posterior_samples[column], bins=50, density=True, color="darkcyan", alpha=0.75)
    ax.axvline(posterior_samples[column].mean(), color="black", linestyle="--", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Density")

plt.tight_layout()
plt.show()


# Forecast probability through 2100
# This is a simple statistical extrapolation, not a physical climate model.
co2_trend_X = sm.add_constant(co2_clean["year"])
co2_trend_model = sm.OLS(co2_clean["co2_ppm"], co2_trend_X).fit()

print(co2_trend_model.summary())

start_forecast_year = int(data["year"].max()) + 1
future_years = np.arange(start_forecast_year, 2101)

future_co2_X = sm.add_constant(future_years)
future_co2 = co2_trend_model.predict(future_co2_X)

future = pd.DataFrame({
    "year": future_years,
    "predicted_CO2": future_co2,
})

print(future.head())
print(future.tail())

future_co2_z = (future["predicted_CO2"].to_numpy() - co2_mean) / co2_std

X_future = np.column_stack([
    np.ones(len(future)),
    future_co2_z,
])

mu_future = beta_samples @ X_future.T
noise = rng.normal(
    loc=0.0,
    scale=np.sqrt(sigma2_samples)[:, None],
    size=mu_future.shape,
)
ice_predictive_samples = mu_future + noise

predicted_ice_mean = ice_predictive_samples.mean(axis=0)
prob_ice_less_than_1 = (ice_predictive_samples < 1.0).mean(axis=0)

forecast_table = pd.DataFrame({
    "year": future["year"],
    "predicted_CO2": future["predicted_CO2"],
    "predicted_ice_mean": predicted_ice_mean,
    "P_ice_less_than_1": prob_ice_less_than_1,
})

print(forecast_table)


# Plot forecast probability
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    forecast_table["year"],
    forecast_table["P_ice_less_than_1"],
    color="purple",
    linewidth=2.5,
)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1.5, label="0.5 probability")
ax.set_xlabel("Year")
ax.set_ylabel("P(sea ice extent < 1 million sq km)")
ax.set_title("Posterior Predictive Probability of Very Low September Sea Ice")
ax.set_ylim(-0.02, 1.02)
ax.legend()
plt.show()


# Report the first crossing year
crossing = forecast_table.loc[forecast_table["P_ice_less_than_1"] > 0.5]

if crossing.empty:
    print("P(Ice_t < 1) does not exceed 0.5 by 2100 in this forecast.")
else:
    first_year = int(crossing.iloc[0]["year"])
    first_prob = crossing.iloc[0]["P_ice_less_than_1"]
    print(f"First year with P(Ice_t < 1) > 0.5: {first_year}")
    print(f"Estimated probability in that year: {first_prob:.3f}")
