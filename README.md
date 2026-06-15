# 🌫️ City Air Quality Forecaster

An end-to-end machine learning project that forecasts hourly AQI (Air Quality Index) for three major Indian cities — **Delhi, Mumbai, and Lucknow** — using real historical data, weather features, and XGBoost time-series modelling. Deployed as an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![XGBoost](https://img.shields.io/badge/XGBoost-✓-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🔗 Live Demo

👉 **[View the live app](YOUR_STREAMLIT_LINK_HERE)**

---

## 📸 Screenshots

| Forecast page | Seasonal heatmap | Model performance |
|---|---|---|
| ![forecast](screenshots/forecast.png) | ![heatmap](screenshots/heatmap.png) | ![model](screenshots/model.png) |

---

## 📌 Problem Statement

Air pollution is one of India's most pressing public health challenges. Delhi consistently ranks among the world's most polluted cities, with AQI spiking dangerously during winter months due to crop burning, cold air trapping particulates, and vehicular emissions.

This project builds a system that:
- Ingests 5 years of real hourly AQI data for Delhi, Mumbai, and Lucknow
- Engineers meaningful predictive features (lag features, weather, Indian festival dates)
- Trains and compares three forecasting models
- Serves a 7-day AQI forecast through an interactive web dashboard

---

## 📊 Dataset

**Source:** [Air Quality Data in India — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)

| City | Rows | Date range | AQI range |
|---|---|---|---|
| Delhi | 47,568 | Jan 2015 – Jul 2020 | 12 – 999 |
| Mumbai | 18,065 | May 2018 – Jul 2020 | 29 – 317 |
| Lucknow | 45,037 | Mar 2015 – Jul 2020 | 20 – 761 |

**Pollutants available:** PM2.5, PM10, NO2, CO, AQI

**Additional data:** Hourly weather (temperature, humidity, wind speed) fetched from [Open-Meteo Historical API](https://open-meteo.com/) — free, no API key required.

---

## 🏗️ Project Structure

```
City Air Quality Forecaster/
│
├── data/                          # Cleaned CSVs per city
│   ├── Delhi_clean.csv
│   ├── Delhi_features.csv
│   ├── Mumbai_clean.csv
│   ├── Mumbai_features.csv
│   ├── Lucknow_clean.csv
│   └── Lucknow_features.csv
│
├── data_collection.py             # Week 1: Load & clean Kaggle dataset
├── 02_eda_features.py             # Week 2: Feature engineering + EDA
├── 03_forecasting.py              # Week 3: Train & evaluate 3 models
├── app.py                         # Week 4: Streamlit dashboard
│
├── Delhi_xgb_model.pkl            # Saved XGBoost models
├── Mumbai_xgb_model.pkl
├── Lucknow_xgb_model.pkl
│
├── model_comparison.png           # MAE/RMSE comparison chart
├── forecast_vs_actual.png         # Predicted vs actual overlay
├── feature_importance.png         # XGBoost feature importance
├── seasonal_decomposition.png     # Trend + seasonal + residual
├── correlation_heatmap.png        # Feature correlations
│
├── city_hour.csv                  # Raw Kaggle dataset (download separately)
└── requirements.txt
```

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `lag_1h`, `lag_3h`, `lag_6h` | AQI value 1, 3, 6 hours ago |
| `lag_12h`, `lag_24h`, `lag_48h` | AQI value 12, 24, 48 hours ago |
| `rolling_24h_mean` | 24-hour rolling average AQI |
| `rolling_24h_std` | 24-hour rolling standard deviation |
| `rolling_7d_mean` | 7-day rolling average AQI |
| `hour`, `day_of_week`, `month` | Time-based cyclical features |
| `is_weekend` | Binary flag for weekends |
| `is_diwali` | Binary flag for Diwali dates (2015–2020) |
| `temperature` | Hourly temperature (°C) from Open-Meteo |
| `humidity` | Relative humidity (%) from Open-Meteo |
| `wind_speed` | Wind speed (km/h) from Open-Meteo |

> **Key insight:** Lag features (especially `lag_1h`) are the strongest predictors of AQI — air quality changes gradually, so the previous hour's reading is highly predictive of the next.

---

## 🤖 Models

Three models were trained and compared using a strict **time-based train/test split** (85%/15%) — never a random shuffle, which would cause data leakage in time-series problems.

| Model | MAE | RMSE | MAPE | Notes |
|---|---|---|---|---|
| **Linear Regression** | **2.03** | **2.78** | **1.2%** | ✅ Best overall — lag features are highly linear |
| XGBoost | 2.66 | 5.18 | 1.5% | ✅ Strong, captures non-linear interactions |
| Prophet | 69.69 | 84.79 | 47.7% | ❌ Underfit — designed for daily/weekly data, not hourly |

**Why Linear Regression won:** When lag features are included, the relationship between past and future AQI is largely linear. XGBoost adds value for non-linear pollutant interactions but the lag signal dominates. Prophet was designed for slower-moving daily/weekly patterns and struggles with hourly autocorrelation.

---

## 📱 Dashboard Features

**Page 1 — Forecast**
- 30-day historical AQI trend
- 7-day rolling XGBoost forecast with ±15% uncertainty band
- CPCB health category colour bands (Good / Satisfactory / Moderate / Poor / Very Poor / Severe)
- 4 live metric cards: current AQI, health category, 7-day avg, 30-day avg

**Page 2 — Seasonal patterns**
- Hour × month heatmap (reveals India's winter pollution spike)
- Day-of-week AQI pattern bar chart

**Page 3 — Model performance**
- MAE/RMSE comparison across all 3 models
- Predicted vs actual overlay chart (30-day window)
- XGBoost feature importance (top 15 features)
- Results summary table with plain-English verdict

---

## 🚀 Run locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/air-quality-forecaster.git
cd air-quality-forecaster
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `city_hour.csv` from [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and place it in the project root.

### 4. Run the pipeline
```bash
python data_collection.py      # Clean & save city CSVs
python 02_eda_features.py      # Build features + fetch weather
python 03_forecasting.py       # Train all 3 models
streamlit run app.py           # Launch dashboard
```

Open **http://localhost:8501** in your browser.

---

## 📦 Requirements

```
streamlit
plotly
pandas
numpy
requests
statsmodels
xgboost
prophet
joblib
matplotlib
seaborn
scikit-learn
```

Install all at once:
```bash
pip install streamlit plotly pandas numpy requests statsmodels xgboost prophet joblib matplotlib seaborn scikit-learn
```

---

## 💡 Key learnings

- **Never shuffle time-series data** for train/test splits — always split chronologically to prevent data leakage
- **Lag features are powerful** — AQI at t-1 is the single most predictive feature, making even a linear model highly accurate
- **Prophet underperforms on hourly data** — it's optimised for daily/weekly business time-series, not high-frequency environmental data
- **Real-world APIs change** — OpenAQ v2 was deprecated in Jan 2025; the project was adapted to use Kaggle's static dataset, a common real-world skill
- **Weather matters** — wind speed and humidity have strong negative correlation with AQI (high wind disperses pollutants)

---

## 🌍 India-specific context

- **Diwali flag:** AQI in Delhi spikes 2–3× during Diwali due to fireworks — modelled as a binary feature
- **Winter spike:** Nov–Jan AQI is consistently worst due to stubble burning in Punjab/Haryana + cold still air trapping particulates — clearly visible in the seasonal heatmap
- **CPCB scale:** Dashboard uses India's Central Pollution Control Board AQI scale, not the US EPA scale

---

## 👤 Author

**Nikhil**
- 📧 [nikhilsaklani7@gmail.com]
- 💼 [https://www.linkedin.com/in/nikhil-saklani/]
- 🐙 [https://github.com/SneaKy17]

---

## 📄 License

MIT License — free to use, modify and distribute.

---

*Built as part of a Data Science portfolio. Dataset credit: [Rohan Rao on Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india).*