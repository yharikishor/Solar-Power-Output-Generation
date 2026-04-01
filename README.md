# ☀️ Solar Panel Power Prediction — Streamlit App

> **Live ML Dashboard** | Predicts solar energy output from environmental variables  
> Group 05 · DS_P599 · Regression Project

---

## 🚀 Quick Deploy to Streamlit Cloud

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - Solar Power Prediction App"
git remote add origin https://github.com/<your-username>/solar-power-prediction.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy!**

---

## 📁 File Structure

```
solar-power-prediction/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── solarpowergeneration.csv      # Dataset (add this!)
├── .streamlit/
│   └── config.toml               # Dark theme configuration
└── README.md
```

> ⚠️ **Important:** Add `solarpowergeneration.csv` to the repo root before deploying.

---

## 🖥️ App Pages

| Page | Description |
|---|---|
| 🏠 Overview | Dataset stats, raw preview, pre-computed results |
| 📊 EDA | Distributions, correlation heatmap, boxplots, scatter plots |
| 🤖 Model Training | Train all 4 models live, compare metrics, diagnostic plots |
| 🔮 Predict | Input environmental values → get power output prediction |

---

## 🤖 Models Included

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **Gradient Boosting** ⭐ | 0.9051 | 3,106 | 1,633 |
| XGBoost | 0.9032 | 3,137 | 1,509 |
| Random Forest | 0.8875 | 3,383 | 1,527 |
| Decision Tree | 0.8143 | 4,346 | 1,913 |

---

## 🛠️ Run Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/solar-power-prediction.git
cd solar-power-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Tech Stack

`Python` · `Streamlit` · `scikit-learn` · `XGBoost` · `pandas` · `matplotlib` · `seaborn`

---

## 👥 Team

**Group 05** — DS_P599 Solar Panel Regression Project
