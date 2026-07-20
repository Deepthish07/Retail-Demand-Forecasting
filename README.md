# 🛍️ Retail Demand Forecasting & Inventory Optimization

> Production-grade Machine Learning pipeline for retail demand forecasting using historical sales data, feature engineering, CatBoost/XGBoost models, and an extensible forecasting architecture.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Forecasting-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# 📌 Project Overview

Retail businesses often struggle with:

- Stock-outs
- Overstocking
- Poor inventory planning
- Seasonal demand fluctuations
- Store-wise demand variations

This project builds a **production-ready retail demand forecasting system** that predicts future sales for every **Store × Style** combination using historical transaction data.

Unlike notebook-based forecasting projects, this repository follows a **real-world software engineering structure** with modular preprocessing, feature engineering, validation, model training, and deployment.

---

# 🎯 Objectives

- Build an end-to-end forecasting pipeline
- Clean raw ERP transaction data
- Automatically detect dataset schemas
- Generate high-quality forecasting features
- Train machine learning forecasting models
- Predict future sales
- Enable inventory optimization
- Deploy as a reusable forecasting application

---

# 🏗 Project Architecture

```
Retail-Demand-Forecasting/

│
├── config/
│   ├── config.yaml
│   ├── column_mapping.yaml
│   └── validation_rules.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── master/
│
├── notebooks/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── test_preprocessing.py
├── test_feature_engineering.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

Retail ERP transaction dataset

### Input Columns

| Column |
|---------|
| Date |
| Store |
| Style |
| Category |
| Description |
| Qty |
| Rate |
| Net Amount |
| Channel |

Current Dataset Size

```
Raw Transactions

590,631 rows
```

After aggregation

```
284,761 daily sales records
```

Calendar completion

```
1,497,805 forecasting records
```

---

# 🚀 Pipeline

```
Raw ERP Data
        │
        ▼
Column Detection
        │
        ▼
Cleaning
        │
        ▼
Business Validation
        │
        ▼
Daily Aggregation
        │
        ▼
Modeling Dataset
        │
        ▼
Calendar Completion
        │
        ▼
Calendar Validation
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Forecast Generation
        │
        ▼
Inventory Recommendation
```

---

# ✅ Implemented Modules

## Data Preprocessing

✔ Automatic column detection

✔ Column renaming

✔ Missing value handling

✔ Text standardization

✔ Business validation

✔ Daily aggregation

✔ Product master creation

✔ Modeling dataset generation

---

## Calendar Completion

Generates continuous daily sales history for every

```
Store × Style
```

Missing dates are automatically inserted.

Example

Before

| Date | Qty |
|------|----:|
|1 Jul|5|
|3 Jul|7|

After

| Date | Qty |
|------|----:|
|1 Jul|5|
|2 Jul|0|
|3 Jul|7|

---

## Feature Engineering

Implemented

✔ Lag Features

```
lag_1
lag_7
lag_14
lag_28
```

Upcoming

```
rolling_mean_7
rolling_mean_28

rolling_sum_7
rolling_sum_28

day_of_week
month
quarter
weekend

holiday features
promotion features
```

---

# 🤖 Machine Learning Models

Planned Models

- CatBoost Regressor ⭐
- XGBoost Regressor
- LightGBM
- Random Forest

Evaluation Metrics

- MAE
- RMSE
- MAPE
- WAPE
- SMAPE

---

# 📈 Forecast Levels

Forecasts can be generated at

- Company Level
- State Level
- Store Level
- Category Level
- Style Level
- Store × Style Level

---

# 🧠 Feature Engineering Strategy

The forecasting model learns from

### Historical Features

- Lag 1
- Lag 7
- Lag 14
- Lag 28

### Rolling Statistics

- 7-day average
- 28-day average
- 7-day rolling sum
- 28-day rolling sum

### Calendar Features

- Day of Week
- Month
- Quarter
- Weekend
- Year

### Business Features

- Store
- Style
- Category
- Channel

---

# 📦 Future Features

- Weather integration
- Festival calendar
- Promotion effects
- Price elasticity
- Inventory optimization
- Safety stock calculation
- Reorder point prediction
- ABC analysis
- XYZ analysis

---

# 📊 Forecast Workflow

```
Historical Sales
        │
        ▼
Feature Engineering
        │
        ▼
CatBoost Training
        │
        ▼
Model Evaluation
        │
        ▼
Future Dates
        │
        ▼
Demand Forecast
        │
        ▼
Inventory Recommendation
```

---

# 📁 Configuration Driven

Project behavior is controlled through YAML files.

```
config.yaml

column_mapping.yaml

validation_rules.yaml
```

No code modification required for different datasets.

---

# 💻 Tech Stack

Programming

- Python

Libraries

- Pandas
- NumPy
- Scikit-Learn
- CatBoost
- XGBoost
- Prophet
- Matplotlib

Deployment

- Streamlit

Version Control

- Git
- GitHub

---

# 📌 Current Progress

| Module | Status |
|---------|---------|
| Data Cleaning | ✅ |
| Validation | ✅ |
| Aggregation | ✅ |
| Product Master | ✅ |
| Calendar Completion | ✅ |
| Calendar Validation | ✅ |
| Lag Features | ✅ |
| Rolling Features | 🚧 |
| Date Features | 🚧 |
| Model Training | 🚧 |
| Hyperparameter Tuning | 🚧 |
| Forecast Generation | 🚧 |
| Streamlit App | 🚧 |

---

# 🚀 Future Roadmap

- Complete Feature Engineering
- Train CatBoost
- Hyperparameter Optimization
- Time Series Cross Validation
- Multi-step Forecasting
- Explainability using SHAP
- Model Registry
- MLflow Integration
- Docker Deployment
- Streamlit Dashboard
- REST API
- CI/CD Pipeline

---

# 📸 Planned Dashboard

The Streamlit application will allow users to

- Upload sales data
- Generate forecasts
- Download predictions
- View demand trends
- Analyze store performance
- Compare historical vs predicted sales

---

# 👨‍💻 Author

**Deepthish Raj**

Artificial Intelligence & Data Science

Retail Analytics | Machine Learning | Forecasting | Data Science

---

# ⭐ Acknowledgements

Inspired by real-world retail forecasting challenges involving ERP transaction data, inventory optimization, and production-grade machine learning pipelines.
