# 02_eda_features.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os

os.makedirs("data", exist_ok=True)

CITIES = ["Delhi", "Mumbai", "Lucknow"]

COORDS = {
    "Delhi":   (28.6139, 77.2090),
    "Mumbai":  (19.0760, 72.8777),
    "Lucknow": (26.8467, 80.9462),
}

DIWALI_DATES = ["2015-11-11", "2016-10-30", "2017-10-19",
                "2018-11-07", "2019-10-27", "2020-11-14"]

def fetch_weather(city_coords, start, end):
    """Open-Meteo HISTORICAL API — works for past dates."""
    lat, lon = city_coords
    # Historical archive endpoint (covers 1940–present)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     "temperature_2m,relativehumidity_2m,windspeed_10m",
        "timezone":   "Asia/Kolkata",
    }
    try:
        r = requests.get(url, params=params, timeout=60).json()
        if "hourly" not in r:
            print(f"  Weather API error: {r.get('reason', 'unknown')}")
            return pd.DataFrame()
        df = pd.DataFrame({
            "datetime":    r["hourly"]["time"],
            "temperature": r["hourly"]["temperature_2m"],
            "humidity":    r["hourly"]["relativehumidity_2m"],
            "wind_speed":  r["hourly"]["windspeed_10m"],
        })
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.set_index("datetime")
    except Exception as e:
        print(f"  Weather fetch failed: {e}")
        return pd.DataFrame()

def build_features(city):
    print(f"\nBuilding features for {city}...")

    df = pd.read_csv(f"data/{city}_clean.csv",
                     index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows")

    # Time features
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_diwali"]   = df.index.normalize().isin(
                            pd.to_datetime(DIWALI_DATES)).astype(int)

    # Lag features
    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"lag_{lag}h"] = df["aqi"].shift(lag)

    # Rolling features
    df["rolling_24h_mean"] = df["aqi"].shift(1).rolling(24).mean()
    df["rolling_24h_std"]  = df["aqi"].shift(1).rolling(24).std()
    df["rolling_7d_mean"]  = df["aqi"].shift(1).rolling(168).mean()

    # Weather — historical API
    start = df.index.min().strftime("%Y-%m-%d")
    end   = df.index.max().strftime("%Y-%m-%d")
    print(f"  Fetching weather: {start} → {end}")
    weather = fetch_weather(COORDS[city], start, end)

    if not weather.empty:
        df = df.join(weather, how="left")
        print(f"  Weather features added ✓")
    else:
        # Add empty columns so shape stays consistent
        df["temperature"] = np.nan
        df["humidity"]    = np.nan
        df["wind_speed"]  = np.nan
        print(f"  Proceeding without weather features")

    # Drop NaN only on lag/rolling cols (NOT on weather cols)
    lag_cols = [f"lag_{l}h" for l in [1,3,6,12,24,48]] + \
               ["rolling_24h_mean", "rolling_24h_std", "rolling_7d_mean"]
    df = df.dropna(subset=["aqi"] + lag_cols)

    out = f"data/{city}_features.csv"
    df.to_csv(out)
    print(f"  Saved {out} — shape: {df.shape}")
    return df

# Build features for all cities
for city in CITIES:
    build_features(city)

# ── Seasonal decomposition — Delhi ────────────────────────────────────────────
print("\nPlotting seasonal decomposition...")
from statsmodels.tsa.seasonal import seasonal_decompose

delhi = pd.read_csv("data/Delhi_clean.csv",
                    index_col=0, parse_dates=True)["aqi"]
daily = delhi.dropna().resample("D").mean().dropna()

result = seasonal_decompose(daily, model="additive", period=365)
fig = result.plot()
fig.set_size_inches(14, 8)
plt.suptitle("Delhi AQI — Seasonal Decomposition", y=1.01)
plt.tight_layout()
plt.savefig("seasonal_decomposition.png", bbox_inches="tight")
print("✓ Saved seasonal_decomposition.png")

# ── Correlation heatmap ───────────────────────────────────────────────────────
print("Plotting correlation heatmap...")
delhi_feat = pd.read_csv("data/Delhi_features.csv",
                          index_col=0, parse_dates=True)

# Only numeric cols with actual data (drop all-NaN cols)
numeric_cols = delhi_feat.select_dtypes(include=np.number).dropna(axis=1, how="all")

plt.figure(figsize=(13, 9))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Delhi — feature correlation matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
print("✓ Saved correlation_heatmap.png")
print("\n✓ EDA complete. Run 03_forecasting.py next.")