# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

st.set_page_config(page_title="India AQI Forecaster", page_icon="🌫️", layout="wide")

CITIES = ["Delhi", "Mumbai", "Lucknow"]

# CPCB India AQI health bands
AQI_BANDS = [
    (0,   50,  "Good",           "#00E400"),
    (51,  100, "Satisfactory",   "#92D050"),
    (101, 200, "Moderate",       "#FFFF00"),
    (201, 300, "Poor",           "#FF7E00"),
    (301, 400, "Very Poor",      "#FF0000"),
    (401, 500, "Severe",         "#7E0023"),
]

def get_aqi_band(value):
    for lo, hi, label, color in AQI_BANDS:
        if lo <= value <= hi:
            return label, color
    return "Severe", "#7E0023"

@st.cache_data
def load_city(city):
    return pd.read_csv(f"data/{city}_features.csv",
                       index_col=0, parse_dates=True)

@st.cache_resource
def load_model(city):
    return joblib.load(f"{city}_xgb_model.pkl")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🌫️ India AQI Forecaster")
city = st.sidebar.selectbox("Select city", CITIES)
page = st.sidebar.radio("View", ["Forecast", "Seasonal patterns", "Model performance"])
pollutant = st.sidebar.selectbox("Pollutant", ["PM2.5", "PM10", "NO2"])

df    = load_city(city)
model = load_model(city)
cols  = pd.read_csv(f"{city}_feature_cols.csv").iloc[:, 0].tolist()

# Current AQI card
latest_aqi   = float(df["aqi"].iloc[-1])
label, color = get_aqi_band(latest_aqi)

st.title(f"Air Quality — {city}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current AQI (PM2.5)", f"{latest_aqi:.0f} µg/m³")
c2.metric("Health category", label)
c3.metric("7-day avg", f"{df['aqi'].iloc[-168:].mean():.0f} µg/m³")
c4.metric("30-day avg", f"{df['aqi'].iloc[-720:].mean():.0f} µg/m³")

st.markdown(f"<div style='padding:8px 16px;background:{color}22;border-left:4px solid {color};border-radius:4px;font-size:13px;color:var(--color-text-primary)'><strong>{label}</strong> — based on India CPCB AQI scale</div>", unsafe_allow_html=True)
st.divider()

# ── Page 1: Forecast ────────────────────────────────────────────────────────
if page == "Forecast":
    st.subheader(f"30-day AQI trend + 7-day forecast")

    last_30d = df["aqi"].iloc[-720:]
    pred_input = df[cols].iloc[-1:]
    future_preds = []

    # Rolling 7-day prediction
    temp_df = df.copy()
    for h in range(168):
        row   = temp_df[cols].iloc[-1:]
        pred  = model.predict(row)[0]
        future_preds.append(pred)
        new_row = temp_df.iloc[-1:].copy()
        new_row.index = [new_row.index[0] + pd.Timedelta(hours=1)]
        new_row["aqi"] = pred
        for lag in [1, 3, 6, 12, 24, 48]:
            if f"lag_{lag}h" in new_row.columns:
                new_row[f"lag_{lag}h"] = temp_df["aqi"].iloc[-lag] if lag <= len(temp_df) else pred
        temp_df = pd.concat([temp_df, new_row])

    future_index = pd.date_range(df.index[-1], periods=169, freq="h")[1:]
    forecast_s   = pd.Series(future_preds, index=future_index)
    uncertainty  = forecast_s * 0.15  # ±15% uncertainty band

    fig = go.Figure()

    # AQI health bands as background
    for lo, hi, lbl, clr in AQI_BANDS:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=clr, opacity=0.07, line_width=0)

    # Historical line
    fig.add_trace(go.Scatter(x=last_30d.index, y=last_30d.values,
                             name="Historical", line=dict(color="#185FA5", width=1.5)))

    # Forecast + uncertainty band
    fig.add_trace(go.Scatter(
        x=list(future_index) + list(future_index[::-1]),
        y=list(forecast_s + uncertainty) + list((forecast_s - uncertainty)[::-1]),
        fill="toself", fillcolor="rgba(29,158,117,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Uncertainty band", showlegend=True
    ))
    fig.add_trace(go.Scatter(x=future_index, y=future_preds,
                             name="7-day forecast",
                             line=dict(color="#1D9E75", width=2, dash="dot")))

    fig.update_layout(height=420, margin=dict(t=30, b=20),
                      legend=dict(orientation="h", y=1.05),
                      yaxis_title="PM2.5 (µg/m³)")
    st.plotly_chart(fig, use_container_width=True)

# ── Page 2: Seasonal patterns ────────────────────────────────────────────────
elif page == "Seasonal patterns":
    st.subheader("AQI seasonal heatmap — hour × month")

    pivot = df.groupby([df.index.month, df.index.hour])["aqi"].mean().unstack()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of day", y="Month", color="Avg PM2.5"),
        y=month_names,
        color_continuous_scale="YlOrRd",
        aspect="auto",
        title=f"{city} — average PM2.5 by hour & month"
    )
    fig.update_layout(height=420, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("🔴 Dark red = high pollution. Notice the Nov–Jan winter spike — a signature of Indian cities due to crop burning and cold, still air trapping particulates.")

# ── Page 3: Model performance ────────────────────────────────────────────────
elif page == "Model performance":
    st.subheader("Model comparison")
    col1, col2 = st.columns(2)
    col1.image("model_comparison.png",  width="stretch")
    col2.image("forecast_vs_actual.png", width="stretch")

    st.subheader("Feature importance (XGBoost)")
    fi = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=True).tail(15)
    fig = px.bar(fi, orientation="h",
                 labels={"index": "Feature", "value": "Importance"},
                 color=fi.values, color_continuous_scale="Teal")
    fig.update_layout(height=400, showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)