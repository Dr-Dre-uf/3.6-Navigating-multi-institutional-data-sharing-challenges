import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import psutil
import os

# --- MONITORING UTILITY ---
def display_performance_monitor():
    """Tracks CPU and RAM usage of the current Streamlit process."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Health")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%")
    c2.metric("Memory", f"{mem_mb:.1f} MB")

# --- APP CONFIG ---
st.set_page_config(page_title="Multi-Institutional Data Sharing (NVFlare Sim)", layout="wide")

# --- SIDEBAR & SETUP ---
st.sidebar.header("Scenario Controls")
track_choice = st.sidebar.radio("Select Research Track:", ["Clinical (IC3 COVID-19)", "Basic Science (ImmPort)"])

st.sidebar.info(
    "**Microskill 6:** Navigate challenges in developing AI-ready datasets for "
    "collaborative research (Federated Learning) while preserving privacy."
)
display_performance_monitor()

# --- TITLE & CONTEXT ---
st.title("🛡️ Navigating Multi-Institutional Data Sharing Challenges")
st.markdown(f"### Current Track: **{track_choice}**")

if track_choice == "Clinical (IC3 COVID-19)":
    st.markdown("""
    * **Context:** Harmonizing patient data from multiple hospitals to predict COVID-19 severity without sharing PHI.
    * **Dataset:** Simulating [IC3 UF Public COVID-19](https://ic3.center.ufl.edu/research/resources/datasets/)
    * **Tool:** Simulating [NVIDIA FLARE](https://nvidia.github.io/NVFlare/) for Federated Learning.
    """)
else:
    st.markdown("""
    * **Context:** Aggregating immunology data from different labs to analyze cytokine responses without moving raw samples.
    * **Dataset:** Simulating [ImmPort](https://www.immport.org/shared/home)
    * **Tool:** Simulating [NVIDIA FLARE](https://nvidia.github.io/NVFlare/) for Federated Learning.
    """)

st.markdown("---")

# --- DATA GENERATION ---
@st.cache_data
def generate_data(track, n_samples=200):
    np.random.seed(42)
    
    if track == "Clinical (IC3 COVID-19)":
        # Simulating Patient Data
        data = {
            'Patient_ID': [f"PT-{i:03d}" for i in range(n_samples)],
            'Institution': np.random.choice(['City_General', 'Mountain_View_Clinic'], n_samples),
            'Age': np.random.randint(18, 90, n_samples),
            # White Blood Cell Count (normal 4500-11000) with outliers
            'WBC_Count': np.append(np.random.normal(7000, 2000, int(n_samples * 0.95)), 
                                   [50000, 60000, 150, 200, 250000] * int(n_samples * 0.01)),
            # C-Reactive Protein (inflammatory marker)
            'CRP_Level': np.random.uniform(0.5, 20.0, n_samples),
            # Oxygen Saturation with missing values
            'O2_Saturation': [np.nan if i % 10 == 0 else x for i, x in enumerate(np.random.normal(96, 2, n_samples))]
        }
    else:
        # Simulating Immunology Lab Data
        data = {
            'Sample_ID': [f"SMP-{i:03d}" for i in range(n_samples)],
            'Institution': np.random.choice(['Lab_Alpha', 'Lab_Beta'], n_samples),
            'Cell_Viability_%': np.random.normal(90, 5, n_samples),
            # Cytokine IL-6 levels (pg/mL) with outliers
            'Cytokine_IL6': np.append(np.random.gamma(2, 10, int(n_samples * 0.95)), 
                                      [500, 600, 800, 1000, 1200] * int(n_samples * 0.01)),
            # Antibody Titer
            'Antibody_Titer': np.random.uniform(100, 5000, n_samples),
            # Flow Cytometry Count with missing values
            'T_Cell_Count': [np.nan if i % 10 == 0 else x for i, x in enumerate(np.random.normal(1200, 300, n_samples))]
        }
    
    # Adjust lengths to match exactly if rounding caused issues (simple truncation)
    min_len = min([len(v) for v in data.values()])
    data = {k: v[:min_len] for k, v in data.items()}
    
    return pd.DataFrame(data)

# Generate and display raw data
df_raw = generate_data(track_choice)

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("1. Raw Local Data")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.caption("Raw data containing outliers, missing values, and institutional identifiers.")

with col2:
    st.subheader("Data Distribution Visualization")
    target_col = 'WBC_Count' if track_choice == "Clinical (IC3 COVID-19)" else 'Cytokine_IL6'
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(x=df_raw[target_col], color='salmon', ax=ax)
    ax.set_title(f"Distribution of {target_col} (Notice Outliers)")
    st.pyplot(fig)

# --- STEP 2: OUTLIER DETECTION (Z-SCORE) ---
st.markdown("### 2. Outlier Detection & Removal")
st.write("Outliers can skew Federated Learning models. We use Z-Score > 3 to identify measurement errors.")

target_col = 'WBC_Count' if track_choice == "Clinical (IC3 COVID-19)" else 'Cytokine_IL6'

# Calculate Z-scores
df_clean = df_raw.copy()
df_clean['z_score'] = zscore(df_clean[target_col])
outliers = df_clean[np.abs(df_clean['z_score']) > 3]

c1, c2 = st.columns(2)
with c1:
    st.metric("Total Rows", len(df_raw))
with c2:
    st.metric("Outliers Detected (Z > 3)", len(outliers))

if st.button("Remove Outliers"):
    df_clean = df_clean[np.abs(df_clean['z_score']) <= 3].drop(columns=['z_score'])
    st.success(f"Removed {len(outliers)} outliers. Dataset is cleaner.")
else:
    st.warning("Click button above to clean data before proceeding.")
    df_clean = df_raw.copy() # Keep raw if not clicked

# --- STEP 3: PREPROCESSING (IMPUTATION & SCALING) ---
st.markdown("### 3. Preprocessing (Imputation & Scaling)")

missing_col = 'O2_Saturation' if track_choice == "Clinical (IC3 COVID-19)" else 'T_Cell_Count'

# Imputation Strategy
impute_method = st.selectbox("Choose Imputation Method for Missing Data:", ["Mean", "Median", "Zero"])

if impute_method == "Mean":
    val = df_clean[missing_col].mean()
elif impute_method == "Median":
    val = df_clean[missing_col].median()
else:
    val = 0

df_clean[missing_col] = df_clean[missing_col].fillna(val)

# Standardization (Scaling)
scaler = MinMaxScaler()
num_cols = df_clean.select_dtypes(include=[np.number]).columns
df_normalized = df_clean.copy()
df_normalized[num_cols] = scaler.fit_transform(df_clean[num_cols])

st.dataframe(df_normalized.head(), use_container_width=True)
st.caption(f"Data Imputed ({impute_method}) and Min-Max Scaled (0-1). Ready for AI Model.")

# --- STEP 4: FEDERATED LEARNING SIMULATION ---
st.markdown("---")
st.subheader("4. Federated Learning Simulation (NVIDIA FLARE Concept)")
st.info("Instead of centralizing the data (unsafe), we split data by Institution and compute 'Local Updates'. We then average the updates to create a Global Model.")

# Split Data
inst_names = df_normalized['Institution'].unique()
inst_A_data = df_normalized[df_normalized['Institution'] == inst_names[0]]
inst_B_data = df_normalized[df_normalized['Institution'] == inst_names[1]]

# Simulate "Training" (Calculating average feature weights as a proxy for model weights)
# In real ML, this would be gradients or neural network weights
features = [c for c in num_cols if c != 'z_score']

weights_A = inst_A_data[features].mean()
weights_B = inst_B_data[features].mean()

# Federated Aggregation
global_weights = (weights_A + weights_B) / 2

# Visualization of the FL Process
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(f"#### 🏥 {inst_names[0]}")
    st.write("**Local Model Weights:**")
    st.dataframe(weights_A, use_container_width=True)
    st.caption("Computed locally. Raw data stays here.")

with col_b:
    st.markdown(f"#### 🏥 {inst_names[1]}")
    st.write("**Local Model Weights:**")
    st.dataframe(weights_B, use_container_width=True)
    st.caption("Computed locally. Raw data stays here.")

with col_c:
    st.markdown("#### 🌐 Global Model")
    st.write("**Aggregated Weights:**")
    st.dataframe(global_weights, use_container_width=True)
    st.success("✅ Improved model derived without sharing patient rows!")

# --- CONCLUSION ---
st.markdown("---")
st.markdown("""
**Key Takeaways:**
1.  **Local Cleaning:** Each institution must handle outliers (Z-score) and missing values locally before training.
2.  **Standardization:** Data must be on the same scale (MinMax) across institutions for the Global Model to make sense.
3.  **Privacy:** By sharing *weights* (middle column) instead of *rows* (top dataframe), we respect privacy and DUA agreements.
""")
