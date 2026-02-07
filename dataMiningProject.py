# ==========================================================
# DOH COVID-19 PHILIPPINES
# Full Regional Impact & Predictive Analysis with Streamlit
# ==========================================================

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PH COVID-19 Regional Analysis", layout="wide")
st.title("🇵🇭 COVID-19 Regional Impact & Risk Analysis (Philippines)")
st.caption("Source: DOH COVID-19 Data Drop – Nov 19, 2022")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "14ff6134-9b08-4252-b98d-e1b0c2d644c0.csv",
        usecols=["RegionRes", "DateDied"],
        low_memory=False
    )

df = load_data()
st.success(f"Dataset loaded successfully: {df.shape[0]:,} records")

# -------------------------------
# CLEAN DATA
# -------------------------------
df["RegionRes"] = df["RegionRes"].astype(str).str.strip().str.upper()
df = df[df["RegionRes"] != "NAN"].dropna(subset=["RegionRes"])
df["DateDied"] = pd.to_datetime(df["DateDied"], errors="coerce")
df["IsDead"] = df["DateDied"].notna()

# -------------------------------
# AGGREGATE DATA
# -------------------------------
summary = (
    df.groupby("RegionRes")
    .agg(TotalCases=("RegionRes", "count"),
         TotalDeaths=("IsDead", "sum"))
    .reset_index()
    .sort_values("TotalCases", ascending=False)
)

# Percentage of national cases
total_cases = summary["TotalCases"].sum()
summary["CasePercentage"] = (summary["TotalCases"] / total_cases * 100).round(2)

# Rank by total cases
summary.insert(0, "Rank", range(1, len(summary) + 1))

# -------------------------------
# PREDICTIVE MODEL: "SAFETY FIRST" SETTINGS
# -------------------------------
# Define High Risk: top 30% of cases
threshold_val = summary["TotalCases"].quantile(0.7)
summary["HighRisk"] = (summary["TotalCases"] >= threshold_val).astype(int)

X = summary[["TotalCases", "TotalDeaths"]]
y = summary["HighRisk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Added 'class_weight' to help the AI focus more on the High Risk category
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)

# 2. SAFETY ADJUSTMENT: Instead of 50% certainty, we lower the bar to 30%
# This helps eliminate False Negatives (Missed Dangers)
custom_threshold = 0.3
probs_test = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (probs_test >= custom_threshold).astype(int)

# Apply this safety threshold to the entire summary
probs_all = model.predict_proba(scaler.transform(X))[:, 1]
summary["PredictedRisk"] = (probs_all >= custom_threshold).astype(int)
summary["PredictedRiskLabel"] = summary["PredictedRisk"].map({1: "High Risk", 0: "Low Risk"})
# -------------------------------
# 1️⃣ TOP 10 MOST AFFECTED REGIONS (CASES)
# -------------------------------
st.header("📌 Top 10 Most Affected Regions by COVID-19 Cases")
top10_cases = summary.head(10)
st.dataframe(top10_cases[["Rank", "RegionRes", "TotalCases", "CasePercentage"]], use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="TotalCases", y="RegionRes", data=top10_cases, palette="Blues_r", ax=ax)
ax.set_title("Top 10 Regions by COVID-19 Cases")
ax.set_xlabel("Total Cases")
ax.set_ylabel("Region")
st.pyplot(fig)

# -------------------------------
# 2️⃣ MOST AFFECTED REGION
# -------------------------------
st.header("🚨 Most Affected Region by COVID-19")
most_affected = summary.iloc[0]
col1, col2, col3 = st.columns(3)
col1.metric("Region", most_affected["RegionRes"])
col2.metric("Total Cases", f"{most_affected['TotalCases']:,}")
col3.metric("Share of National Cases", f"{most_affected['CasePercentage']}%")

# -------------------------------
# 3️⃣ TOP 10 REGIONS WITH MOST DEATHS
# -------------------------------
st.header("⚠️ Top 10 Regions with Highest COVID-19 Deaths")
top10_deaths = summary.sort_values("TotalDeaths", ascending=False).head(10).copy()
top10_deaths["DeathRank"] = range(1, len(top10_deaths) + 1)
st.dataframe(top10_deaths[["DeathRank", "RegionRes", "TotalDeaths"]], use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x="TotalDeaths", y="RegionRes", data=top10_deaths, palette="Reds_r", ax=ax)
ax.set_title("Top 10 Regions by COVID-19 Deaths")
ax.set_xlabel("Total Deaths")
ax.set_ylabel("Region")
st.pyplot(fig)

# -------------------------------
# 4️⃣ FULL REGIONAL TABLE (OVERALL)
# -------------------------------
st.header("📋 Overall COVID-19 Regional Summary")
st.dataframe(summary[["Rank", "RegionRes", "TotalCases", "CasePercentage", "TotalDeaths"]], use_container_width=True)

# -------------------------------
# 5️⃣ DISPLAY PREDICTED RISK
# -------------------------------
st.subheader("Predicted Risk for All Regions")

# Create a display-specific dataframe to avoid modifying the main 'summary'
risk_display = summary[["RegionRes", "TotalCases", "TotalDeaths", "PredictedRisk", "PredictedRiskLabel"]].copy()

st.dataframe(
    risk_display.sort_values("PredictedRisk", ascending=False)
    .drop(columns=["PredictedRisk"]), # Hide the numeric column after sorting
    use_container_width=True
)
st.info("High Risk regions are predicted based on total cases and deaths using Logistic Regression.")
# -------------------------------
# GEOGRAPHIC RISK MAP
# -------------------------------
#st.header("🗺️ Geographic Risk Distribution")

# 1. Coordinate Dictionary for PH Regions (Approximate Centers)
#region_coords = {
  #  "NCR": [14.5995, 120.9842],
  #  "REGION IV-A (CALABARZON)": [14.1008, 121.0794],
 #   "REGION III (CENTRAL LUZON)": [15.4828, 120.7120],
  #  "REGION VII (CENTRAL VISAYAS)": [10.3157, 123.8854],
  #  "REGION VI (WESTERN VISAYAS)": [10.7202, 122.5621],
  #  "REGION XI (DAVAO REGION)": [7.1907, 125.4553],
    # ... you can add more regions here!
#}

# 2. Map the coordinates to your summary dataframe
#map_df = summary.copy()
#map_df['lat'] = map_df['RegionRes'].map(lambda x: region_coords.get(x, [None, None])[0])
#map_df['lon'] = map_df['RegionRes'].map(lambda x: region_coords.get(x, [None, None])[1])

# 3. Clean up any regions we didn't provide coordinates for
#map_df = map_df.dropna(subset=['lat', 'lon'])

# 4. Display the map
# High Risk regions will show up prominently if you filter them
#st.map(map_df, latitude='lat', longitude='lon', size='TotalCases', color='#ff4b4b' if any(map_df['PredictedRisk'] == 1) else '#0000ff')

#st.info("💡 The bubbles represent the location of each region. Larger bubbles indicate more cases.")
# -------------------------------
# 6️⃣ MODEL METRICS
# -------------------------------
st.subheader("Model Performance Metrics")
accuracy = accuracy_score(y_test, y_pred)
st.markdown(f"- **Accuracy:** {accuracy:.2f}")
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report_df)

# -------------------------------
# 6️⃣ MODEL METRICS & LEGEND
# -------------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ADD THIS LEGEND BLOCK:
with st.expander("📖 How to read this Matrix (Legend)"):
    st.markdown("""
    The Confusion Matrix shows how many regions the model classified correctly versus where it made mistakes:

    * **Top-Left (True Negative):** Low-risk regions correctly identified as **Low Risk**.
    * **Bottom-Right (True Positive):** High-risk regions correctly identified as **High Risk**.
    * **Top-Right (False Positive):** 'False Alarms'—Low-risk regions wrongly flagged as **High Risk**.
    * **Bottom-Left (False Negative):** 'Missed Targets'—High-risk regions wrongly labeled as **Low Risk**.
    """)

    # Optional: Display the specific counts dynamically
    tn, fp, fn, tp = cm.ravel()
    st.write(f"**Current Results:** Correctly Identified: {tn + tp} | Mistakes: {fp + fn}")

# -------------------------------
# 8️⃣ DOWNLOAD REPORT
# -------------------------------
st.header("📥 Export Data")

@st.cache_data
def convert_df(df_to_convert):
    # This converts the dataframe into a CSV format for download
    return df_to_convert.to_csv(index=False).encode('utf-8')

# We'll export the full summary including your perfect predictions
csv_data = convert_df(summary)

st.download_button(
    label="Download Regional Risk Report (CSV)",
    data=csv_data,
    file_name="PH_COVID_Risk_Analysis.csv",
    mime="text/csv",
)
st.success("You can now download the full analysis for use in Excel or other reporting tools.")
# -------------------------------
# 7️⃣ KEY OBSERVATIONS
# -------------------------------
st.header("🧾 Key Observations")
st.markdown("""
- COVID-19 cases are concentrated in a few regions; the National Capital Region typically leads.
- Some regions with fewer cases may still have high mortality rates, indicating healthcare disparities.
- Predictive modeling identifies high-risk regions for proactive health interventions.
- Rankings, percentages, and predictions help policymakers prioritize resources.
""")
st.info("Percentages represent each region’s contribution to total confirmed COVID-19 cases nationwide.")
