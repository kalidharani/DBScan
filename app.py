import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances_argmin_min

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="DBSCAN Cluster Finder",
    page_icon="🧩",
    layout="centered"
)

st.title("🍷 Wine DBSCAN Clustering App")

# -------------------------------
# Load Model, Scaler, Data
# -------------------------------
@st.cache_resource
def load_all():
    dbscan = joblib.load("wine_dbscan_model.pkl")
    scaler = joblib.load("wine_scaler.pkl")
    data = pd.read_csv("wine_clustering_data.csv")
    return dbscan, scaler, data

dbscan, scaler, data = load_all()

# -------------------------------
# Detect Feature Names
# -------------------------------
if hasattr(scaler, "feature_names_in_"):
    feature_names = list(scaler.feature_names_in_)
else:
    feature_names = data.columns.tolist()

# -------------------------------
# User Input Section (2 Columns)
# -------------------------------
st.subheader("🔢 Enter Feature Values")

col1, col2 = st.columns(2)
user_input = []

for i, feature in enumerate(feature_names):
    default_value = float(data[feature].mean())

    if i % 2 == 0:
        value = col1.number_input(feature, value=default_value)
    else:
        value = col2.number_input(feature, value=default_value)

    user_input.append(value)

input_array = np.array(user_input).reshape(1, -1)

# -------------------------------
# Predict Cluster
# -------------------------------
if st.button("Find Cluster"):
    try:
        # Scale input
        scaled_input = scaler.transform(input_array)

        # DBSCAN nearest core logic
        core_samples = dbscan.components_
        core_labels = dbscan.labels_[dbscan.core_sample_indices_]

        closest_core, distance = pairwise_distances_argmin_min(
            scaled_input, core_samples
        )

        cluster = core_labels[closest_core[0]]

        # -------------------------------
        # Output
        # -------------------------------
        if cluster == -1:
            st.error("❌ This data point is classified as **NOISE (Outlier)**")
        else:
            st.success(f"✅ This data point belongs to **Cluster {cluster}**")

        st.caption("Note: DBSCAN assigns label -1 to noise points.")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")