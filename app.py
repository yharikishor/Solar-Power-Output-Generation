import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="☀️ Solar Power Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main-title {
        font-size: 2.6rem; font-weight: 700;
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center;
    }
    .metric-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { color: #f8fafc; font-size: 1.8rem; font-weight: 700; }
    .metric-sub   { color: #64748b; font-size: 0.75rem; }

    .section-header {
        font-size: 1.3rem; font-weight: 600; color: #f1f5f9;
        border-left: 4px solid #f59e0b; padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    .best-badge {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        color: white; padding: 0.2rem 0.7rem; border-radius: 99px;
        font-size: 0.75rem; font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    'distance-to-solar-noon', 'temperature', 'wind-direction',
    'wind-speed', 'sky-cover', 'visibility', 'humidity',
    'average-wind-speed-(period)', 'average-pressure-(period)'
]
TARGET = 'power-generated'

MODEL_RESULTS = {
    "Gradient Boosting": {"R2": 0.9051, "RMSE": 3106.00, "MAE": 1633.37},
    "XGBoost":           {"R2": 0.9032, "RMSE": 3137.03, "MAE": 1509.21},
    "Random Forest":     {"R2": 0.8875, "RMSE": 3382.84, "MAE": 1526.75},
    "Decision Tree":     {"R2": 0.8143, "RMSE": 4345.73, "MAE": 1912.67},
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data
def preprocess(df):
    df = df.copy()
    df['average-wind-speed-(period)'].fillna(df['average-wind-speed-(period)'].mean(), inplace=True)
    df.drop_duplicates(inplace=True)

    def cap_outliers(series):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return series.clip(lo, hi)

    for col in df.select_dtypes(['int', 'float']).columns:
        df[col] = cap_outliers(df[col])
    return df

@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Decision Tree":     DecisionTreeRegressor(random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0),
    }

    results, predictions = {}, {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        predictions[name] = (y_test, pred)
        results[name] = {
            "R2":   round(r2_score(y_test, pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, pred)), 2),
            "MAE":  round(mean_absolute_error(y_test, pred), 2),
        }

    return models, scaler, results, predictions, X_test_s

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sun.png", width=60)
    st.markdown("## ☀️ Solar Power Predictor")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview", "📊 EDA", "🤖 Model Training", "🔮 Predict"])
    st.markdown("---")
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Upload `solarpowergeneration.csv`", type=["csv"])
    st.markdown("---")
    st.caption("DS_P599 · Group 05 · Solar Panel Regression")

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">☀️ Solar Panel Power Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Regression Analysis · Group 05 · DS_P599</div>', unsafe_allow_html=True)

# ── No file fallback ──────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Please upload the `solarpowergeneration.csv` file from the sidebar to begin.")

    st.markdown('<div class="section-header">📈 Pre-computed Model Results</div>', unsafe_allow_html=True)
    res_df = pd.DataFrame(MODEL_RESULTS).T.reset_index().rename(columns={"index": "Model"})
    res_df = res_df.sort_values("R2", ascending=False).reset_index(drop=True)

    cols = st.columns(4)
    for i, row in res_df.iterrows():
        badge = " 🏆" if i == 0 else ""
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{row['Model']}{badge}</div>
                <div class="metric-value">{row['R2']:.4f}</div>
                <div class="metric-sub">R² Score</div>
                <div class="metric-sub">RMSE: {row['RMSE']:,.0f} &nbsp;|&nbsp; MAE: {row['MAE']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.stop()

# ── Load & preprocess ─────────────────────────────────────────────────────────
df_raw  = load_data(uploaded)
df      = preprocess(df_raw)

# ════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Rows</div><div class="metric-value">{df_raw.shape[0]:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Features</div><div class="metric-value">{len(FEATURES)}</div></div>', unsafe_allow_html=True)
    with c3:
        missing = int(df_raw.isnull().sum().sum())
        st.markdown(f'<div class="metric-card"><div class="metric-label">Missing Values</div><div class="metric-value">{missing}</div></div>', unsafe_allow_html=True)
    with c4:
        dups = int(df_raw.duplicated().sum())
        st.markdown(f'<div class="metric-card"><div class="metric-label">Duplicates</div><div class="metric-value">{dups}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">📋 Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.markdown('<div class="section-header">📐 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.describe().round(3), use_container_width=True)

    st.markdown('<div class="section-header">🏆 Model Performance Summary</div>', unsafe_allow_html=True)
    res_df = pd.DataFrame(MODEL_RESULTS).T.reset_index().rename(columns={"index": "Model"})
    res_df = res_df.sort_values("R2", ascending=False).reset_index(drop=True)
    cols = st.columns(4)
    for i, row in res_df.iterrows():
        badge = " 🏆" if i == 0 else ""
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{row['Model']}{badge}</div>
                <div class="metric-value">{row['R2']:.4f}</div>
                <div class="metric-sub">R² Score</div>
                <div class="metric-sub">RMSE: {row['RMSE']:,.0f}</div>
                <div class="metric-sub">MAE: {row['MAE']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="section-header">📊 Feature Distributions</div>', unsafe_allow_html=True)
    num_cols = FEATURES + [TARGET]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.patch.set_facecolor('#0f172a')
    for ax, col in zip(axes.flatten(), num_cols):
        sns.histplot(df[col], kde=True, ax=ax, color='#f59e0b', edgecolor='none', alpha=0.8)
        ax.set_title(col, color='white', fontsize=9)
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor('#334155')
    for ax in axes.flatten()[len(num_cols):]: ax.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-header">🔗 Correlation Heatmap</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    fig2.patch.set_facecolor('#0f172a')
    ax2.set_facecolor('#1e293b')
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlGn', ax=ax2,
                linewidths=0.5, annot_kws={'size': 8},
                cbar_kws={'shrink': 0.8})
    ax2.tick_params(colors='white', labelsize=9)
    ax2.set_title("Correlation Matrix", color='white', fontsize=14)
    st.pyplot(fig2)

    st.markdown('<div class="section-header">📦 Box Plots (After Outlier Capping)</div>', unsafe_allow_html=True)
    fig3, axes3 = plt.subplots(3, 4, figsize=(16, 9))
    fig3.patch.set_facecolor('#0f172a')
    for ax, col in zip(axes3.flatten(), num_cols):
        sns.boxplot(y=df[col], ax=ax, color='#f59e0b', width=0.5)
        ax.set_title(col, color='white', fontsize=9)
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor('#334155')
    for ax in axes3.flatten()[len(num_cols):]: ax.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown('<div class="section-header">🎯 Scatter: Features vs Power Generated</div>', unsafe_allow_html=True)
    fig4, axes4 = plt.subplots(3, 3, figsize=(14, 10))
    fig4.patch.set_facecolor('#0f172a')
    for ax, col in zip(axes4.flatten(), FEATURES):
        ax.scatter(df[col], df[TARGET], alpha=0.3, s=5, color='#f59e0b')
        ax.set_xlabel(col, color='#94a3b8', fontsize=8)
        ax.set_ylabel('power-generated', color='#94a3b8', fontsize=8)
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=7)
        for spine in ax.spines.values(): spine.set_edgecolor('#334155')
    plt.tight_layout()
    st.pyplot(fig4)

# ════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    with st.spinner("Training all 4 models... this may take a moment ⚙️"):
        models, scaler, results, predictions, _ = train_models(df)

    st.success("✅ All models trained successfully!")

    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    res_df = res_df.sort_values("R2", ascending=False).reset_index(drop=True)

    cols = st.columns(4)
    for i, row in res_df.iterrows():
        badge = " 🏆" if i == 0 else ""
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{row['Model']}{badge}</div>
                <div class="metric-value">{row['R2']:.4f}</div>
                <div class="metric-sub">R²</div>
                <div class="metric-sub">RMSE: {row['RMSE']:,.0f}</div>
                <div class="metric-sub">MAE: {row['MAE']:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    # R2 bar chart
    st.markdown("")
    fig_bar, ax_bar = plt.subplots(figsize=(9, 4))
    fig_bar.patch.set_facecolor('#0f172a')
    ax_bar.set_facecolor('#1e293b')
    colors = ['#f59e0b' if i == 0 else '#475569' for i in range(len(res_df))]
    bars = ax_bar.barh(res_df["Model"], res_df["R2"], color=colors, height=0.5)
    for bar, val in zip(bars, res_df["R2"]):
        ax_bar.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', color='white', fontsize=10)
    ax_bar.set_xlabel("R² Score", color='white')
    ax_bar.set_title("Model Comparison — R² Score", color='white', fontsize=13)
    ax_bar.tick_params(colors='white')
    ax_bar.set_xlim(0, 1.05)
    ax_bar.invert_yaxis()
    for spine in ax_bar.spines.values(): spine.set_edgecolor('#334155')
    st.pyplot(fig_bar)

    # Per-model diagnostic plots
    st.markdown('<div class="section-header">🔍 Diagnostic Plots per Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Select model to inspect", list(predictions.keys()))

    y_test, y_pred = predictions[model_choice]
    residuals = y_test - y_pred

    c1, c2 = st.columns(2)
    with c1:
        fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
        fig_sc.patch.set_facecolor('#0f172a'); ax_sc.set_facecolor('#1e293b')
        ax_sc.scatter(y_test, y_pred, alpha=0.4, s=10, color='#f59e0b')
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax_sc.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
        ax_sc.set_xlabel("Actual", color='white'); ax_sc.set_ylabel("Predicted", color='white')
        ax_sc.set_title(f"{model_choice}: Actual vs Predicted", color='white')
        ax_sc.tick_params(colors='#94a3b8')
        for sp in ax_sc.spines.values(): sp.set_edgecolor('#334155')
        st.pyplot(fig_sc)

    with c2:
        fig_res, ax_res = plt.subplots(figsize=(6, 4))
        fig_res.patch.set_facecolor('#0f172a'); ax_res.set_facecolor('#1e293b')
        ax_res.scatter(y_pred, residuals, alpha=0.4, s=10, color='#38bdf8')
        ax_res.axhline(0, color='red', linestyle='--', lw=1.5)
        ax_res.set_xlabel("Predicted", color='white'); ax_res.set_ylabel("Residuals", color='white')
        ax_res.set_title(f"{model_choice}: Residual Plot", color='white')
        ax_res.tick_params(colors='#94a3b8')
        for sp in ax_res.spines.values(): sp.set_edgecolor('#334155')
        st.pyplot(fig_res)

    fig_line, ax_line = plt.subplots(figsize=(13, 4))
    fig_line.patch.set_facecolor('#0f172a'); ax_line.set_facecolor('#1e293b')
    ax_line.plot(y_test.values[:150], label="Actual", color='#f59e0b', lw=1.5)
    ax_line.plot(y_pred[:150], label="Predicted", color='#38bdf8', lw=1.5, alpha=0.85)
    ax_line.set_title(f"{model_choice}: Actual vs Predicted (first 150 samples)", color='white')
    ax_line.set_xlabel("Index", color='white'); ax_line.set_ylabel("Power Generated", color='white')
    ax_line.legend(facecolor='#1e293b', labelcolor='white')
    ax_line.tick_params(colors='#94a3b8')
    for sp in ax_line.spines.values(): sp.set_edgecolor('#334155')
    st.pyplot(fig_line)

# ════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<div class="section-header">🔮 Predict Solar Power Output</div>', unsafe_allow_html=True)
    st.markdown("Enter environmental conditions to predict energy generation using the **Gradient Boosting** model.")

    with st.spinner("Training model..."):
        models, scaler, results, predictions, _ = train_models(df)

    gb_model = models["Gradient Boosting"]

    stats = df[FEATURES].describe()

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        inputs = {}
        feat_cols = [c1, c2, c3, c1, c2, c3, c1, c2, c3]
        for feat, col in zip(FEATURES, feat_cols):
            with col:
                mn = float(stats.loc['min', feat])
                mx = float(stats.loc['max', feat])
                med = float(stats.loc['50%', feat])
                inputs[feat] = st.number_input(feat, min_value=mn, max_value=mx, value=med, format="%.4f")

        submitted = st.form_submit_button("⚡ Predict Power Output")

    if submitted:
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)
        prediction = gb_model.predict(input_scaled)[0]

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f59e0b22,#ef444422);
                    border:1px solid #f59e0b; border-radius:16px;
                    padding:2rem; text-align:center; margin-top:1.5rem;">
            <div style="color:#94a3b8;font-size:0.9rem;letter-spacing:0.1em;text-transform:uppercase;">
                Predicted Power Generated
            </div>
            <div style="font-size:3.5rem;font-weight:800;
                        background:linear-gradient(135deg,#f59e0b,#ef4444);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                {prediction:,.0f} W
            </div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:0.5rem;">
                Gradient Boosting · R² = 0.9051
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        with st.expander("📋 Input Summary"):
            st.dataframe(pd.DataFrame([inputs]).T.rename(columns={0: "Value"}))
