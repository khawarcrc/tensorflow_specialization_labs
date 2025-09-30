# =====================================================
# 📘 Time Series Analysis Lab — Synthetic Data Tutorial
# =====================================================
# This lab walks through synthetic time series generation, 
# visualization, autocorrelation, autoregressive processes, 
# and ARIMA forecasting.
#
# We build:
# 1. Trend, Seasonality, Noise (basic building blocks)
# 2. Synthetic series combinations
# 3. Autoregressive processes (AR)
# 4. ACF/PACF analysis
# 5. ARIMA model fitting and forecasting
# 6. Decomposition of time series
#
# Every plot is explained in detail.
# =====================================================


# -------------------------
# 0. Import Libraries
# -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
np.random.seed(42)  # for reproducibility of random numbers


# -------------------------
# Helper Function: Plotting
# -------------------------
def plot_series(time_index, values, title="", ylabel="Value"):
    """Utility to plot a time series with proper labels and title."""
    plt.figure(figsize=(10, 4))              # set figure size
    plt.plot(time_index, values, label="series")  # plot the line
    plt.title(title, fontsize=14)            # add a descriptive title
    plt.xlabel("Time")                       # x-axis = time
    plt.ylabel(ylabel)                       # y-axis = value
    plt.grid(alpha=0.25)                     # light grid for easier reading
    plt.tight_layout()                       # adjust layout
    plt.show()                               # show the plot


# =====================================================
# 1. Basic Building Blocks of a Time Series
# =====================================================

def make_trend(length, slope=0.0):
    """Return a linear trend of given length and slope."""
    return slope * np.arange(length)  # slope * time index = linear growth


def seasonal_pattern(season_time):
    """Return a custom seasonal pattern for demonstration.
    
    - Uses a cosine wave for smooth oscillation.
    - Adds a localized spike to simulate unusual events (e.g., promotions, holidays).
    """
    pattern = np.cos(2 * np.pi * season_time)  # cosine = smooth cycles
    pattern += 0.3 * np.exp(-30 * (season_time - 0.45) ** 2)  # spike around mid-season
    return pattern


def make_seasonality(length, period, amplitude=1.0):
    """Generate a repeating seasonal signal with given period and amplitude."""
    season_time = (np.arange(length) % period) / period  # fractional cycle position
    return amplitude * np.array([seasonal_pattern(t) for t in season_time])


def white_noise(length, noise_level=1.0):
    """Generate Gaussian white noise with mean 0 and given standard deviation."""
    return np.random.normal(loc=0.0, scale=noise_level, size=length)


# =====================================================
# 2. Generate Synthetic Series
# =====================================================

# Time index: 400 daily data points
length = 400
time_index = pd.date_range(start="2020-01-01", periods=length, freq="D")

# 2.A: Pure Trend
trend = make_trend(length, slope=0.05)
plot_series(time_index, trend, title="Pure Linear Trend (slope=0.05)")
# 🔎 Explanation: A straight upward line; slope defines steepness.


# 2.B: Pure Seasonality
season = make_seasonality(length, period=30, amplitude=2.0)
plot_series(time_index, season, title="Pure Seasonality (period=30, amplitude=2)")
# 🔎 Explanation: Repeating cycle every ~30 days, representing monthly-like seasonality.


# 2.C: Trend + Seasonality
series_ts = trend + season
plot_series(time_index, series_ts, title="Trend + Seasonality (additive)")
# 🔎 Explanation: Seasonal pattern oscillates around an upward-moving baseline (trend).


# 2.D: Add Noise
noisy_series = series_ts + white_noise(length, noise_level=1.0)
plot_series(time_index, noisy_series, title="Trend + Seasonality + Noise")
# 🔎 Explanation: Random fluctuations obscure the clear seasonal cycle,
# making real-world analysis harder.


# =====================================================
# 3. Autocorrelation and AR Processes
# =====================================================

# 3.A: Simulate AR(1):  x_t = phi * x_{t-1} + e_t
phi = 0.8
ar1 = np.zeros(length)                        # buffer for AR(1) series
epsilon = white_noise(length, noise_level=1.0)  # white noise shocks
for t in range(1, length):
    ar1[t] = phi * ar1[t - 1] + epsilon[t]    # recursive AR(1) formula

plot_series(time_index, ar1, title=f"AR(1) process (phi={phi})")
# 🔎 Explanation: Strong persistence; values are correlated with their immediate past.


# 3.B: ACF and PACF Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ar1, lags=30, ax=axes[0])    # autocorrelation function
plot_pacf(ar1, lags=30, ax=axes[1])   # partial autocorrelation function
plt.tight_layout()
plt.show()
# 🔎 Explanation:
# - ACF: slowly decays for AR(1), showing persistence.
# - PACF: shows a sharp cut-off after lag 1 (signature of AR(1)).


# =====================================================
# 4. ARIMA Model Fitting & Forecasting
# =====================================================

# Fit ARIMA(1,0,0) to the AR(1) series
model = ARIMA(ar1, order=(1, 0, 0))
arima_fit = model.fit()

# Forecast next 30 days
forecast = arima_fit.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot actual series + forecast
plt.figure(figsize=(12, 5))
plt.plot(time_index, ar1, label="Observed")
plt.plot(
    pd.date_range(time_index[-1], periods=30, freq="D"),
    forecast_mean,
    color="red",
    label="Forecast"
)
plt.fill_between(
    pd.date_range(time_index[-1], periods=30, freq="D"),
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink",
    alpha=0.3,
    label="Confidence Interval"
)
plt.title("ARIMA(1,0,0) Forecast of AR(1) Process")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
# 🔎 Explanation:
# - The model captures AR(1) persistence well.
# - Forecast reverts toward mean with uncertainty growing further out.


# =====================================================
# 5. Seasonal Decomposition
# =====================================================

# Decompose noisy synthetic series (additive assumption)
decomp = seasonal_decompose(noisy_series, model="additive", period=30)

# Plot decomposition
decomp.plot()
plt.suptitle("Seasonal Decomposition of Synthetic Series", fontsize=14)
plt.tight_layout()
plt.show()
# 🔎 Explanation:
# - Trend: gradual upward slope.
# - Seasonal: repeating cycle extracted.
# - Residual: remaining noise/randomness.
