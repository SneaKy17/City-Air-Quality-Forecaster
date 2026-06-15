# data_collection.py  — uses Kaggle India AQI dataset (no API needed)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Download the dataset manually (one time only)
# Go to: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
# Click Download → extract → you get "city_day.csv" and "city_hour.csv"
# Place "city_hour.csv" inside your project folder
# ─────────────────────────────────────────────────────────────────────────────

RAW_FILE = "city_hour.csv"   # file from Kaggle

if not os.path.exists(RAW_FILE):
    print("ERROR: city_hour.csv not found.")
    print("Download it from: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india")
    exit()

print("Loading raw dataset...")
df = pd.read_csv(RAW_FILE, parse_dates=["Datetime"])
print(f"  Raw shape: {df.shape}")
print(f"  Columns:   {df.columns.tolist()}")
print(f"  Cities:    {df['City'].unique().tolist()}")

# ── Filter to 3 Indian cities ─────────────────────────────────────────────────
CITIES = ["Delhi", "Mumbai", "Lucknow"]
df = df[df["City"].isin(CITIES)].copy()
print(f"\nAfter city filter: {df.shape}")

# ── Rename & select columns ───────────────────────────────────────────────────
# Dataset has: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, AQI
df = df.rename(columns={
    "Datetime": "datetime",
    "City":     "city",
    "PM2.5":    "pm25",
    "PM10":     "pm10",
    "NO2":      "no2",
    "CO":       "co",
    "AQI":      "aqi",
})

KEEP_COLS = ["datetime", "city", "pm25", "pm10", "no2", "co", "aqi"]
df = df[KEEP_COLS].copy()

# ── Clean per city ────────────────────────────────────────────────────────────
for city in CITIES:
    city_df = df[df["city"] == city].copy()
    city_df = city_df.set_index("datetime").sort_index()
    city_df = city_df.drop(columns=["city"])

    # Resample to hourly (fills any missing timestamps)
    city_df = city_df.resample("1h").mean()

    # Forward-fill small gaps only (≤ 3 hours)
    city_df = city_df.interpolate(method="time", limit=3)

    # Drop rows where AQI is still missing after interpolation
    city_df = city_df.dropna(subset=["aqi"])

    out_path = f"data/{city}_clean.csv"
    city_df.to_csv(out_path)
    print(f"\n{city}: {len(city_df)} hourly rows → saved to {out_path}")
    print(f"  Date range: {city_df.index.min()} → {city_df.index.max()}")
    print(f"  AQI range:  {city_df['aqi'].min():.0f} – {city_df['aqi'].max():.0f}")

# ── Data availability heatmap ─────────────────────────────────────────────────
matrix, labels = [], []
MAX_MONTHS = 36

for city in CITIES:
    path = f"data/{city}_clean.csv"
    if not os.path.exists(path):
        continue
    s = pd.read_csv(path, index_col=0, parse_dates=True)["aqi"]
    monthly = s.resample("ME").count() / (24 * 30)
    
    # Pad or trim to exactly MAX_MONTHS columns so all rows are equal length
    vals = monthly.values
    if len(vals) < MAX_MONTHS:
        vals = np.pad(vals, (0, MAX_MONTHS - len(vals)), constant_values=0)
    else:
        vals = vals[:MAX_MONTHS]
    
    matrix.append(vals)
    labels.append(city)

if matrix:
    fig, ax = plt.subplots(figsize=(14, 3))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Month (up to 36 months)")
    plt.colorbar(im, label="Data availability")
    plt.title("Hourly AQI data availability by city")
    plt.tight_layout()
    plt.savefig("data_availability.png", bbox_inches="tight")
    print("✓ Saved data_availability.png")