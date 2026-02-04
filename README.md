# Train Auxiliary Energy Consumption Modeling

A machine learning project for modeling and predicting the energy consumed by auxiliary equipment in railway trains. The analysis combines linear regression with interpretable coefficients and Random Forest for comparison, using real operational data from train HVAC systems.

![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Regression-green.svg)
![Domain](https://img.shields.io/badge/Domain-Railway-red.svg)

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Methods](#methods)
- [Key Findings](#key-findings)
- [Results](#results)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Author](#author)

## Overview

This project models the energy consumption in the AC branch of train auxiliary converters based on the operating times of HVAC (Heating, Ventilation, and Air Conditioning) components. The goal is to estimate power consumption coefficients (in kW) for each component, enabling energy auditing and predictive maintenance.

**Business Context:** The auxiliary equipment consuming from the AC branch includes air production compressors and HVAC systems for both passenger compartments and driver cabins. Understanding their individual contributions to total energy consumption is crucial for operational efficiency.

## Dataset

One month of operational data (September 2019) aggregated by event type:

| Variable | Description | Units |
|----------|-------------|-------|
| `date_time` | Timestamp (YYYY-MM-DD HH:MM:SS) | - |
| `ut` | Train unit identifier | - |
| `eaux_train_ac` | Energy consumed by AC branch (**Target**) | kWh |
| `event` | Event type (interstation/stop/maintenance) | - |
| `event_duration` | Event duration | seconds |
| `suma_ton_compresores` | Total ON-time of air compressors | seconds |
| `suma_hvac_ton_compresores` | Total ON-time of saloon HVAC compressors | seconds |
| `suma_hvac_ton_heater` | Total ON-time of saloon HVAC heaters | seconds |
| `suma_ton_comp_cab` | Total ON-time of cabin HVAC compressors | seconds |
| `suma_ton_heater_cab` | Total ON-time of cabin HVAC heaters | seconds |

**Mathematical Model:**

```
y ≈ β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

Where:
- `y`: AC auxiliary energy per event [kWh]
- `x₁..ₙ`: Operating times of auxiliary components [s]
- `β₁..ₙ`: Average power estimated by the model [kW after conversion]

## Methods

### Data Preprocessing
- Filter events > 600 seconds (outliers)
- Remove maintenance events
- Calculate power (kW) from energy and duration

### Statistical Analysis
- Shapiro-Wilk normality tests
- Wilcoxon rank-sum test for group comparisons
- Cliff's Delta effect size
- ROC/AUC analysis for discriminative capacity

### Regression Modeling
- **Linear Regression** with all HVAC component times
- **Stratified Analysis** by event type (interstation vs stop)
- **Multicollinearity Check** via correlation matrix
- **Residual Diagnostics** including QQ-plots and homoscedasticity tests

### Machine Learning Comparison
- **Random Forest Regressor** with feature importance analysis
- RMSE comparison between linear and ensemble approaches

## Key Findings

### 1. Energy Consumption Differences by Event Type
- **Significant difference** in energy consumption (`eaux_train_ac`) between interstation and stop events
- Cliff's Delta ≈ 0.82 (large effect) → interstation events consume substantially more energy
- AUC ≈ 0.91 → excellent discriminatory capacity

### 2. Power (kW) Differences by Event Type
- **No practical difference** in instantaneous power between event types
- Cliff's Delta ≈ 0.08 (negligible effect)
- AUC ≈ 0.54 → near-random discrimination

### 3. Linear Model Performance
- **R² ≈ 0.993** → explains 99.3% of energy consumption variance
- All predictors highly significant (p < 2e-16)
- Model generalizes well: RMSE_train ≈ 0.062, RMSE_validation ≈ 0.059

### 4. Estimated Power Coefficients (kW)

| Component | Estimated Power |
|-----------|-----------------|
| Air Compressors | 12.67 kW |
| HVAC Compressors (Saloon) | 7.82 kW |
| HVAC Heaters (Saloon) | 9.84 kW |
| HVAC Compressors (Cabin) | 2.39 kW |
| HVAC Heaters (Cabin) | 1.52 kW |
| Event Duration Effect | 12.30 kW |
| Stop Event Premium | +17.26 kW |

### 5. Random Forest vs Linear Regression

| Model | RMSE Train | RMSE Validation |
|-------|------------|-----------------|
| Linear Regression | 0.0620 | 0.0586 |
| Random Forest | 0.1303 | 0.1359 |

**Conclusion:** Linear regression outperforms Random Forest for this problem while maintaining full interpretability of coefficients.

## Results

```
┌─────────────────────────────────────────────────────────────┐
│              LINEAR REGRESSION MODEL                        │
├─────────────────────────────────────────────────────────────┤
│  R² = 0.993  |  RMSE = 0.062 kWh                            │
│  All coefficients significant (p < 2e-16)                   │
│  Excellent generalization (validation RMSE ≈ train RMSE)    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              MODEL SELECTION                                │
├─────────────────────────────────────────────────────────────┤
│  Selected: Linear Regression                                │
│  Reasons:                                                   │
│    - Lower RMSE than Random Forest                          │
│    - Full coefficient interpretability (kW per component)   │
│    - Physical meaning of each parameter                     │
│    - Suitable for energy auditing and reporting             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
train-auxiliary-energy-modeling/
│
├── README.md                    # Project documentation
├── ASSIGNMENT.md               # Original assignment description
│
├── src/
│   └── auxiliary_energy_analysis.R   # Main analysis script
│
└── data/
    └── data.RData       # Dataset (not included)
```
