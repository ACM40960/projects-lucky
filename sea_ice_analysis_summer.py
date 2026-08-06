import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
import numpy as np


# Load the data
september = pd.read_csv("N_09_extent_v4.0.csv")
print(september.head())


# Plot the sea ice extent
plt.plot(september['year'], september[' extent'])
plt.title("Arctic Sea Ice Extent in September (1979-2026)")
plt.xlabel("Year")
plt.ylabel("Extent (10^6 km²)")
plt.show()


# Fit the linear model
# Define X and Y
X = september['year']
Y = september[' extent']

# Add intercept
X = sm.add_constant(X)

# Fit linear regression：Ordinary Least Squares
model = sm.OLS(Y, X)
results = model.fit()

# Show results
print(results.summary())


# Plot the linear trend
# Predicted values from regression
y_pred = results.predict(X)

plt.figure(figsize=(10,6))

# Original data
plt.plot(september['year'],
         september[' extent'],
         label='Observed Extent')

# Regression line
plt.plot(september['year'],
         y_pred,
         color='red',
         linewidth=2,
         label='Linear Trend')

plt.title('September Arctic Sea Ice Extent (1979–2026)——Linear Regression')
plt.xlabel('Year')
plt.ylabel('Extent (10^6 km²)')

plt.legend()

plt.show()


# Ice-free Arctic means September sea ice extent below 1 million km²
X = september['year'].values
Y = september[' extent'].values
t = X - X.min()


# Fit the polynomial model
# Polynomial model
coef2 = np.polyfit(t, Y, deg=2)
print("Polynomial coefficients:", coef2)
poly2 = np.poly1d(coef2)


# Plot the polynomial prediction
future_years = np.arange(X.min(), 2200)
future_t = future_years - X.min()

Y_poly2 = poly2(future_t)

plt.figure(figsize=(8,5))
plt.scatter(X, Y, label='Observed data')
plt.plot(future_years, Y_poly2, label='Quadratic polynomial model')

plt.axhline(y=1, linestyle='--', label='Ice-free threshold = 1')
plt.axhline(y=0, linestyle=':', label='Zero ice')

plt.xlabel('Year')
plt.ylabel('September sea ice extent')
plt.title('Quadratic Polynomial Prediction')
plt.legend()
plt.show()


# Find the first ice-free year from the polynomial model
coef = coef2.copy()

coef[-1] = coef[-1] - 1

roots = np.roots(coef)

valid_root = roots[roots > 0][0]

ice_free_year = 1979 + valid_root

print(ice_free_year)


# Fit the exponential model
# Exponential model
from scipy.optimize import curve_fit

def exp_model(t, A, k):
    return A * np.exp(-k * t)

params, covariance = curve_fit(
    exp_model,
    t,
    Y,
    p0=[Y[0], 0.02]
)

A, k = params

print("A =", A)
print("k =", k)


# Plot the exponential prediction
Y_exp = exp_model(future_t, A, k)

plt.figure(figsize=(8,5))
plt.scatter(X, Y, label='Observed data')
plt.plot(future_years, Y_exp, label='Exponential model')

plt.axhline(y=1, linestyle='--', label='Ice-free threshold = 1')
plt.axhline(y=0, linestyle=':', label='Zero ice')

plt.xlabel('Year')
plt.ylabel('September sea ice extent')
plt.title('Exponential Model Prediction')
plt.legend()
plt.show()


# Check the threshold
below_1_exp = future_years[Y_exp < 1]

if len(below_1_exp) > 0:
    print("Exponential model: first year below 1 million km²:", below_1_exp[0])
else:
    print("Exponential model: not below 1 million km² before 2100")
