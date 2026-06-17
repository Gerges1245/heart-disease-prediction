# 🫀 Heart Disease Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9400D3?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Machine Learning system to predict heart disease risk from lifestyle and medical survey data**  
*October University for Modern Science and Arts — CS363 Machine Learning Project*

[📊 Dataset](#-dataset) • [🧠 Models](#-models) • [📈 Results](#-results) • [🚀 Quick Start](#-quick-start) • [👥 Team](#-team)

</div>

---

## 📌 Overview

Heart disease is the **#1 cause of death worldwide**, claiming **17.9 million lives annually** (WHO). This project builds a clinical decision-support system that predicts an individual's risk of heart disease using lifestyle surveys and medical history — no lab tests required.

| Metric | Value |
|---|---|
| 📦 Dataset Size | 301,717 samples (after deduplication) |
| 🔬 Features | 18 raw → **33 engineered** |
| 🤖 Models Trained | 5 individual + 1 Stacking Ensemble |
| 🏆 Best AUC | **83.51%** (Stacking Ensemble) |
| 🎯 Best Recall | **53.8%** (Stacking Ensemble) |
| ⚖️ Imbalance Fix | SMOTE (9% → 50% positive class) |

---

## 📊 Dataset

**Source:** [CDC BRFSS 2020 — Heart Disease cleaned (Kaggle)](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

| Property | Detail |
|---|---|
| Raw rows | 319,795 |
| After deduplication | **301,717** |
| Train / Test split | 241,373 / 60,344 (80/20) |
| Positive class | 27,261 cases (9.04%) |
| Negative class | 274,456 cases (90.96%) |

### Feature Categories

| Category | Features |
|---|---|
| **Target** | `HeartDisease` |
| **Health Conditions** | Stroke, Asthma, KidneyDisease, SkinCancer, Diabetic |
| **Lifestyle** | Smoking, AlcoholDrinking, PhysicalActivity, SleepTime |
| **Physical & Mental** | BMI, PhysicalHealth (0–30 sick days), MentalHealth (0–30 sick days) |
| **Demographics** | AgeCategory (13 bands), Sex, Race (6 categories), GenHealth, DiffWalking |

---

## ⚙️ Data Preprocessing & Feature Engineering

```
Raw CSV (319,795)
  → Drop Duplicates → 301,717
  → Binary Encoding (Yes/No columns)
  → Ordinal Encoding (Age, GenHealth)
  → One-Hot Encoding (Race, Diabetic)
  → Feature Engineering (+15 new features → 33 total)
  → 80/20 Train-Test Split
  → SMOTE Balancing (Train Only, seed=42)
```

### Engineered Features (15 New)

| Feature | Description |
|---|---|
| `Age_BMI` | AgeCategory × BMI interaction |
| `BMI_sq` | BMI² — captures nonlinear obesity risk |
| `BMI_Obese` | Binary flag: BMI ≥ 30 |
| `Smoking_Age` | Smoking × AgeCategory interaction |
| `HighRisk` | 1 if Stroke OR DiffWalking = 1 |
| `SleepRisk` | 1 if Sleep < 6 hrs or > 9 hrs |
| `Comorbid` | Count of co-occurring conditions |

### SMOTE Balancing
```
BEFORE → 91% No Disease / 9% Heart Disease (241,373 training samples)
AFTER  → 50% No Disease / 50% Heart Disease (439,128 training samples)
⚠️  SMOTE applied ONLY to training set — test set preserves real-world 9% distribution
```

---

## 🧠 Models

### Individual Models

| # | Model | Key Parameters |
|---|---|---|
| 1 | **Logistic Regression** (Baseline) | C=1.0, solver=lbfgs, class_weight='balanced' |
| 2 | **Random Forest** (Bagging) | n_estimators=300, max_depth=14, balanced_subsample |
| 3 | **XGBoost** (Boosting) | scale_pos_weight tuned, threshold=0.72 |
| 4 | **LightGBM** (Boosting) | is_unbalance=True, threshold=0.23 |
| 5 | **HistGradientBoosting** | class_weight='balanced', threshold=0.22 |

### 🏆 Stacking Ensemble Architecture

```
Layer 1 — Base Models (5-Fold OOF + SMOTE per fold)
┌─────────────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────────────┐
│ Logistic Regr.  │  │ Random Forest│  │ XGBoost │  │ LightGBM │  │ HistGradientBoosting  │
└────────┬────────┘  └──────┬───────┘  └────┬────┘  └────┬─────┘  └──────────┬────────────┘
         └──────────────────┴───────────────┴────────────┴────────────────────┘
                           OOF Predictions Matrix [241,373 × 5]
                                           ↓
Layer 2 — Meta-Learner
  Logistic Regression (C=2, class_weight='balanced', StandardScaler)
                                           ↓
                           Best Threshold: 0.73 → Final Predictions
```

---

## 📈 Results

### Model Comparison (Real Test Set — 60,344 samples)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Threshold |
|---|---|---|---|---|---|---|
| Logistic Regression | 85.2% | 31.2% | 52.6% | 39.2% | 0.8310 | 0.70 |
| Random Forest | 85.6% | 31.5% | 50.4% | 38.8% | 0.8290 | 0.34 |
| XGBoost | 85.3% | 31.5% | 53.6% | 39.7% | 0.8334 | 0.72 |
| LightGBM | 86.7% | 33.6% | 48.5% | 39.7% | 0.8329 | 0.23 |
| HistGradientBoosting | 86.0% | 32.5% | 51.7% | 39.9% | 0.8349 | 0.22 |
| ⭐ **Stacking Ensemble** | **85.4%** | **31.8%** | **53.8%** | **40.0%** | **0.8351** | **0.73** |

> **Key Insight:** The Stacking Ensemble achieves the best AUC (83.51%) and highest Recall (53.8%), making it the most suitable model for clinical use where missing a diagnosis is the most costly error.

### Confusion Matrix — Stacking Ensemble

```
                    Predicted: No Disease   Predicted: Heart Disease
Actual: No Disease       TN = 48,600 ✓          FP = 6,292
Actual: Heart Disease    FN = 2,517 ⚠️           TP = 2,935 ✓

Detected (TP):    2,935  (53.8%)
Missed (FN):      2,517  (46.2%)   ← primary area for improvement
False Alarms (FP):6,292  (11.5%)
True Negatives:  48,600  (88.5%)
```

---

## 🔍 Key EDA Findings

- **Age Risk:** Heart disease rises sharply after 55. The 75–79 age band shows ~35% prevalence vs <1% in ages 18–24.
- **BMI Impact:** Obese patients (BMI ≥ 30) are **1.8× more likely** to have heart disease.
- **Smoking:** Smokers show **1.6× higher risk**. Combined with alcohol, risk compounds significantly.
- **Physical Activity:** Inactive individuals have **1.5× higher** disease rate.
- **Comorbidities:** Stroke history is the **strongest individual predictor**. DiffWalking correlates strongly with disease presence.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn ydata-profiling
```

### Run the Notebook
```bash
# Clone the repository
git clone https://github.com/MB1382005/heart-disease-prediction.git
cd heart-disease-prediction

# Download the dataset from Kaggle and place it as:
# heart_2020_cleaned.csv

# Run the notebook
jupyter notebook FinalPart_one_Last_Last.ipynb
```

### Project Structure
```
heart-disease-prediction/
├── FinalPart_one_Last_Last.ipynb   # Main ML pipeline notebook
├── Heart_Deasis_Prediction.html    # EDA profiling report (ydata-profiling)
├── HeartDisease_ML_Enhanced.pptx   # Project presentation
└── README.md
```

---

## 🔮 Future Work

- [ ] **Deep Learning:** TabNet / Neural Networks on tabular data
- [ ] **Explainability:** SHAP values for clinical decision transparency
- [ ] **External Validation:** Cross-validate on more diverse datasets
- [ ] **Hyperparameter Tuning:** Optuna / Bayesian optimization
- [ ] **Deployment:** Real-time clinical decision support API
- [ ] **Reduce FNR:** Missed diagnoses are the most clinically harmful outcome

---


**Instructors:** Assoc. Prof. Nermin Abd El-Wahab · Dr. Ebtsam El-Hussany  
**Course:** Machine Learning CS363 — October University for Modern Science and Arts  
**Submission:** May 11, 2026

---

<div align="center">

</div>
