import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ClimaRisk-Malaria-Surveillance-Dashboard",
    layout="wide"
)

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    region_df = pd.read_csv("data/processed/region_risk_zones.csv")
    df = pd.read_csv("data/processed/advanced_cleaned_data.csv")
    return region_df, df

@st.cache_resource
def load_model():
    return joblib.load("models/tuned_rf_model.pkl")

region_df, df = load_data()
model = load_model()

# ---------------- HEADER ----------------
st.title("🌍 Climate-Driven Malaria Risk System")
st.caption("Early Warning & Decision Support Dashboard")
st.markdown("---")

# ---------------- KPI SECTION ----------------
st.subheader("📊 Key Risk Indicators")

high_risk_df = region_df[region_df["risk_level"] == "High Risk"]

k1, k2, k3, k4 = st.columns(4)

k1.metric("Avg Malaria Cases", round(region_df["malaria_cases"].mean(), 2))
k2.metric("Outbreak Probability", f"{round(region_df['outbreak_flag'].mean()*100,1)}%")
k3.metric("High Risk Regions", high_risk_df["region"].nunique())
k4.metric(
    "Highest Risk Region",
    high_risk_df.sort_values("malaria_cases", ascending=False)["region"].iloc[0]
)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Risk Overview", "🌦️ Climate Analysis", "📆 Trends & Seasonality", "🧪 Custom Graph Builder"]
)

# =========================================================
# TAB 1: RISK OVERVIEW
# =========================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regional Malaria Risk Zones")
        fig = px.bar(
            region_df,
            x="region",
            y="malaria_cases",
            color="risk_level",
            title="Malaria Cases by Region & Risk Level"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 High-Burden Regions")
        top_regions = region_df.sort_values("malaria_cases", ascending=False).head(10)
        fig = px.bar(
            top_regions,
            y="region",
            x="malaria_cases",
            orientation="h",
            title="Top 10 Regions by Malaria Cases"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outbreak Probability by Region")
    outbreak_prob = df.groupby("region")["outbreak_flag"].mean().reset_index()
    fig = px.bar(
        outbreak_prob,
        x="outbreak_flag",
        y="region",
        orientation="h",
        title="Outbreak Probability"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2: CLIMATE ANALYSIS
# =========================================================
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Climate Drivers of Malaria Risk")
        fig = px.scatter(
            region_df,
            x="precipitation_mm",
            y="avg_temp_c",
            size="population_density",
            color="risk_level",
            hover_name="region",
            title="Rainfall vs Temperature"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Healthcare Budget vs Malaria")
        fig = px.scatter(
            df,
            x="healthcare_budget",
            y="malaria_cases",
            trendline="ols",
            title="Healthcare Budget Impact"
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3: TRENDS & SEASONALITY
# =========================================================
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Yearly Malaria Trend")
        yearly = df.groupby("year")["malaria_cases"].mean().reset_index()
        fig = px.line(
            yearly,
            x="year",
            y="malaria_cases",
            markers=True,
            title="Yearly Average Malaria Cases"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Seasonality")
        monthly = df.groupby("month")["malaria_cases"].mean().reset_index()
        fig = px.line(
            monthly,
            x="month",
            y="malaria_cases",
            markers=True,
            title="Monthly Malaria Pattern"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonal Heatmap")
    heatmap_df = df.pivot_table(
        values="malaria_cases",
        index="region",
        columns="month",
        aggfunc="mean"
    )

    fig = px.imshow(
        heatmap_df,
        color_continuous_scale="Reds",
        title="Seasonal Malaria Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4: CUSTOM GRAPH BUILDER
# =========================================================
with tab4:
    st.subheader("Build Your Own Graph")

    numeric_cols = df.select_dtypes(include=np.number).columns

    col1, col2, col3 = st.columns(3)

    x_axis = col1.selectbox("X-axis", numeric_cols)
    y_axis = col2.selectbox("Y-axis", numeric_cols, index=1)
    graph_type = col3.selectbox("Graph Type", ["Scatter", "Line", "Bar"])

    if graph_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    elif graph_type == "Line":
        fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    else:
        fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")

    st.plotly_chart(fig, use_container_width=True)

# ---------------- ML PREDICTION ----------------
st.markdown("---")
st.subheader("🤖 Predict Malaria Risk for New Conditions")

col1, col2, col3 = st.columns(3)

temp = col1.slider("Avg Temperature (°C)", 10.0, 40.0, 25.0)
rain = col1.slider("Rainfall (mm)", 0.0, 500.0, 150.0)
month = col1.selectbox("Month", list(range(1, 13)))

pop = col2.slider("Population Density", 50, 5000, 500)
budget = col2.slider("Healthcare Budget Index", 1000, 50000, 15000)

if st.button("Predict Risk"):
    input_data = np.array([[temp, rain, 100, 5, pop, budget, month, 50, 120]])
    pred = model.predict(input_data)[0]

    st.success(f"Predicted Malaria Cases: {int(pred)}")

    if pred > df["malaria_cases"].quantile(0.85):
        st.error("⚠️ HIGH RISK of Outbreak")
    elif pred > df["malaria_cases"].quantile(0.6):
        st.warning("⚠️ MEDIUM RISK of Outbreak")
    else:
        st.success("✅ LOW RISK of Outbreak")

# ---------------- INSIGHTS ----------------
st.markdown("""
### 🔍 Key Insights
- Rainfall and temperature are dominant malaria drivers  
- High population density amplifies outbreaks  
- Healthcare investment reduces severity, not exposure  

### 🛠 Recommended Actions
- Target high-risk regions before monsoon  
- Strengthen healthcare readiness seasonally  
- Use scenario simulation for early warning  
""")

