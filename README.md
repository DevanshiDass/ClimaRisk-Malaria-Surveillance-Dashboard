# 🌍 ClimaRisk: Climate-Driven Malaria Surveillance Dashboard

An end-to-end **Data Science & Machine Learning project** that analyzes, predicts, and visualizes malaria risk under changing climate conditions using an interactive **Streamlit dashboard**.

---

## 📌 Problem Statement

Climate change plays a critical role in the spread of vector-borne diseases such as malaria.  
Most traditional surveillance systems are reactive and lack predictive intelligence.

This project aims to build a **climate-aware malaria surveillance system** that integrates data science, machine learning, and interactive visualization to support early warning and decision-making.

---

## 🎯 Project Objectives

- Analyze how climate variables influence malaria outbreaks  
- Identify high-risk regions and seasonal patterns  
- Predict malaria case burden under new climate scenarios  
- Provide decision-ready insights through an interactive dashboard  

---

## 🧠 Complete Data Science Pipeline

This project follows a **full, structured data science lifecycle** as outlined below:


### Pipeline Explanation

- **Data Collection**  
  Climate, demographic, and health-related data are collected and stored as CSV files.

- **Data Cleaning & Preprocessing**  
  Missing values, inconsistent formats, and noise are handled in preprocessing notebooks.

- **Feature Engineering**  
  Derived features such as outbreak flags, seasonal indicators, and risk labels are created.

- **Exploratory Data Analysis (EDA)**  
  Statistical summaries and visual exploration are performed to understand patterns and correlations.

- **Time Series & Seasonal Analysis**  
  Monthly and yearly trends are analyzed to capture seasonal malaria behavior.

- **Regional Risk Zoning**  
  Regions are classified into Low, Medium, and High risk categories based on malaria burden and outbreak behavior.

- **Machine Learning Modeling**  
  A Random Forest regression model is trained to predict malaria case burden.

- **Model Evaluation & Tuning**  
  Model performance is evaluated using RMSE and R², with hyperparameter tuning applied.

- **Deployment via Streamlit Dashboard**  
  The trained model and processed data are deployed in an interactive Streamlit dashboard.

---

## 📊 Dashboard Features

### Key Risk Indicators (KPIs)
- Average malaria cases  
- Outbreak probability (frequency-based)  
- Number of high-risk regions  
- Region with highest malaria burden (severity-based)  

### Visual Analytics
- Regional Malaria Risk Zones  
- Climate Drivers (Rainfall vs Temperature)  
- Seasonal Heatmap (Region × Month)  
- Yearly Malaria Trends  
- Monthly Seasonality  
- Outbreak Probability by Region  
- Healthcare Budget vs Malaria Cases  
- Top 10 High-Burden Regions  

All visualizations are interactive and exportable.

---

## 🤖 Machine Learning Component

- **Model:** Random Forest Regressor  
- **Target Variable:** Malaria case burden  
- **Input Features:** Climate, demographic, healthcare, and seasonal variables  
- **Evaluation Metrics:** RMSE, R²  
- **Tuning:** GridSearchCV / RandomizedSearchCV  

The trained model is serialized and loaded into the dashboard for real-time inference.

---

## 🔮 Prediction Simulator

The dashboard includes a simulator allowing users to adjust:
- Average temperature  
- Rainfall  
- Population density  
- Healthcare budget  
- Month of the year  

The model predicts malaria cases and classifies risk as:
- Low  
- Medium  
- High  

---

## 🧩 Technology Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit  
- **Tools:** Git, GitHub, VS Code, Jupyter Notebook  

---


