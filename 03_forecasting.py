# 03_forecasting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from prophet import Prophet

CITY = "Delhi"

print(f"Loading {CITY} features...")
df = pd.read_csv(f"data/{CITY}_features.csv",
                 index_col=0, parse_dates=True)
df = df.dropna(subset=["aqi"])
df = df.dropna(axis=1, how="all")

FEATURE_COLS = [c for c in df.columns if c != "aqi"]

# ── Drop any remaining NaNs in features ──────────────────────────────────────
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
print(f"  NaNs after fill: {df[FEATURE_COLS].isnull().sum().sum()}")

# ── Train/test split ──────────────────────────────────────────────────────────
split = int(len(df) * 0.85)
train, test = df.iloc[:split], df.iloc[split:]
print(f"Train: {len(train)} rows | Test: {len(test)} rows")
# ── Model 1: Linear Regression baseline (replaces SARIMA — faster, still valid)
print("\n[1/3] Training Linear Regression baseline...")
lr = LinearRegression()
lr.fit(train[FEATURE_COLS], train["aqi"])
lr_pred  = lr.predict(test[FEATURE_COLS])
lr_mae   = mean_absolute_error(test["aqi"], lr_pred)
lr_rmse  = np.sqrt(mean_squared_error(test["aqi"], lr_pred))
lr_mape  = np.mean(np.abs((test["aqi"].values - lr_pred)
                           / test["aqi"].values)) * 100
print(f"  MAE={lr_mae:.2f}  RMSE={lr_rmse:.2f}  MAPE={lr_mape:.1f}%")

# ── Model 2: XGBoost ─────────────────────────────────────────────────────────
print("\n[2/3] Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,
    eval_metric="mae",
    random_state=42,
)
xgb_model.fit(
    train[FEATURE_COLS], train["aqi"],
    eval_set=[(test[FEATURE_COLS], test["aqi"])],
    verbose=False,
)
xgb_pred  = xgb_model.predict(test[FEATURE_COLS])
xgb_mae   = mean_absolute_error(test["aqi"], xgb_pred)
xgb_rmse  = np.sqrt(mean_squared_error(test["aqi"], xgb_pred))
xgb_mape  = np.mean(np.abs((test["aqi"].values - xgb_pred)
                             / test["aqi"].values)) * 100
print(f"  MAE={xgb_mae:.2f}  RMSE={xgb_rmse:.2f}  MAPE={xgb_mape:.1f}%")

joblib.dump(xgb_model, f"{CITY}_xgb_model.pkl")
pd.Series(FEATURE_COLS).to_csv(f"{CITY}_feature_cols.csv", index=False)
print(f"  Saved {CITY}_xgb_model.pkl")

# ── Model 3: Prophet ─────────────────────────────────────────────────────────
print("\n[3/3] Training Prophet...")
prophet_train = train[["aqi"]].reset_index()
prophet_train.columns = ["ds", "y"]
prophet_train["ds"] = prophet_train["ds"].dt.tz_localize(None)

prophet = Prophet(
    seasonality_mode="multiplicative",
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    interval_width=0.85,
)
prophet.add_country_holidays(country_name="IN")
prophet.fit(prophet_train)

future   = prophet.make_future_dataframe(periods=len(test), freq="h")
forecast = prophet.predict(future)
prophet_pred = forecast.set_index("ds")["yhat"].iloc[-len(test):].values
prophet_mae  = mean_absolute_error(test["aqi"], prophet_pred)
prophet_rmse = np.sqrt(mean_squared_error(test["aqi"], prophet_pred))
prophet_mape = np.mean(np.abs((test["aqi"].values - prophet_pred)
                               / test["aqi"].values)) * 100
print(f"  MAE={prophet_mae:.2f}  RMSE={prophet_rmse:.2f}  MAPE={prophet_mape:.1f}%")

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "="*52)
print(f"{'Model':<22} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
print("-"*52)
print(f"{'Linear Regression':<22} {lr_mae:>8.2f} {lr_rmse:>8.2f} {lr_mape:>7.1f}%")
print(f"{'XGBoost':<22} {xgb_mae:>8.2f} {xgb_rmse:>8.2f} {xgb_mape:>7.1f}%")
print(f"{'Prophet':<22} {prophet_mae:>8.2f} {prophet_rmse:>8.2f} {prophet_mape:>7.1f}%")
print("="*52)

# ── Model comparison chart ────────────────────────────────────────────────────
models = ["Linear Reg.", "XGBoost", "Prophet"]
maes   = [lr_mae,   xgb_mae,  prophet_mae]
rmses  = [lr_rmse,  xgb_rmse, prophet_rmse]
colors = ["#888780", "#1D9E75", "#7F77DD"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, vals, title, ylabel in zip(
    axes,
    [maes, rmses],
    [f"{CITY} — MAE (lower is better)", f"{CITY} — RMSE (lower is better)"],
    ["MAE (AQI units)", "RMSE (AQI units)"]
):
    bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v:.1f}", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches="tight")
print("✓ Saved model_comparison.png")

# ── Predicted vs Actual (30-day window) ──────────────────────────────────────
plot_n  = min(30 * 24, len(test))
actual  = test["aqi"].values[-plot_n:]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for ax, preds, name, color in zip(
    axes,
    [lr_pred[-plot_n:], xgb_pred[-plot_n:], prophet_pred[-plot_n:]],
    ["Linear Regression", "XGBoost", "Prophet"],
    ["#888780", "#1D9E75", "#7F77DD"]
):
    ax.plot(actual, label="Actual", color="#185FA5", lw=1.2, alpha=0.8)
    ax.plot(preds,  label=name,     color=color,     lw=1.2, alpha=0.85)
    ax.set_ylabel("AQI")
    ax.legend(loc="upper right")
    ax.set_title(f"{name} — predicted vs actual")

axes[-1].set_xlabel("Hours")
plt.tight_layout()
plt.savefig("forecast_vs_actual.png", bbox_inches="tight")
print("✓ Saved forecast_vs_actual.png")

# ── Feature importance ────────────────────────────────────────────────────────
fi = pd.Series(xgb_model.feature_importances_,
               index=FEATURE_COLS).sort_values(ascending=True).tail(15)
plt.figure(figsize=(10, 6))
fi.plot(kind="barh", color="#1D9E75")
plt.title(f"{CITY} — XGBoost feature importance (top 15)")
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance.png", bbox_inches="tight")
print("✓ Saved feature_importance.png")

print("\n✓ All done! Run: streamlit run app.py")

for city in ["Mumbai", "Lucknow"]:
    print(f"\nTraining XGBoost for {city}...")
    
    df_city = pd.read_csv(f"data/{city}_features.csv",
                          index_col=0, parse_dates=True)
    df_city = df_city.dropna(subset=["aqi"])
    df_city = df_city.dropna(axis=1, how="all")

    # Use only columns that exist in THIS city's dataframe
    city_feature_cols = [c for c in df_city.columns if c != "aqi"]
    
    # Fill NaNs
    df_city[city_feature_cols] = df_city[city_feature_cols].fillna(
                                  df_city[city_feature_cols].median())

    split   = int(len(df_city) * 0.85)
    train_c = df_city.iloc[:split]
    test_c  = df_city.iloc[split:]

    model_c = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        eval_metric="mae",
        random_state=42,
    )
    model_c.fit(
        train_c[city_feature_cols], train_c["aqi"],
        eval_set=[(test_c[city_feature_cols], test_c["aqi"])],
        verbose=False,
    )

    mae  = mean_absolute_error(test_c["aqi"],
                               model_c.predict(test_c[city_feature_cols]))
    rmse = np.sqrt(mean_squared_error(test_c["aqi"],
                               model_c.predict(test_c[city_feature_cols])))
    print(f"  {city} — MAE={mae:.2f}  RMSE={rmse:.2f}")

    joblib.dump(model_c, f"{city}_xgb_model.pkl")
    pd.Series(city_feature_cols).to_csv(f"{city}_feature_cols.csv", index=False)
    print(f"  Saved {city}_xgb_model.pkl ✓")

print("\n✓ All city models saved. Run: streamlit run app.py")