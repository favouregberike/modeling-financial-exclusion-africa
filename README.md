# Modeling Financial Exclusion Drivers in Africa
**[Live Demo →](https://modeling-financial-exclusion-africa1.streamlit.app/)**

## Project Overview

Financial inclusion is a critical driver of economic participation, poverty reduction, and sustainable development across Africa. Despite the rapid growth of digital finance and mobile money services, millions of people remain excluded from formal financial systems.

This project applies data science and machine learning techniques to analyze financial exclusion patterns across African populations using the World Bank Global Findex dataset.

The objective is to identify the demographic and structural factors most associated with financial exclusion and model exclusion intensity across countries and population groups.

Using exploratory data analysis, feature engineering, predictive modeling, and model explainability techniques, this project generates data-driven insights that can support governments, fintech companies, development organizations, and financial institutions in improving access to financial services.

---

# Business Problem

Financial exclusion limits access to:
- savings,
- credit,
- insurance,
- digital payments,
- and economic opportunities.

Understanding which populations are most financially excluded and why is essential for designing effective financial inclusion strategies.

This project uses machine learning to uncover the strongest predictors of financial exclusion and evaluate how factors such as gender, age, urbanization, and digital finance adoption influence access to financial services across Africa.

---

# Objectives

The objectives of this project are to:

- Analyze demographic patterns of financial exclusion
- Identify key barriers to banking access
- Examine digital financial adoption trends
- Engineer features from socioeconomic indicators
- Build predictive machine learning models
- Evaluate feature importance and model explainability
- Generate actionable business and policy insights

---

# Dataset

The dataset used in this project is the World Bank Global Findex Database.

Due to GitHub file size limitations, the raw dataset is not included in this repository.

You can download the dataset from:

https://www.worldbank.org/en/publication/globalfindex

After downloading:
1. Place the CSV file inside the `data/` folder
2. Update the file path in the notebook if necessary
   
Key dimensions include:
- Country
- Gender
- Age Group
- Urbanization
- Financial Activity Indicators
- Time Period

---

# Data Science Workflow

## 1. Data Cleaning & Preparation

- Filtered relevant financial exclusion indicators
- Handled missing values
- Encoded categorical variables
- Structured data for machine learning workflows
- Engineered analytical features

---

## 2. Exploratory Data Analysis (EDA)

Conducted exploratory analysis to identify:
- country-level exclusion patterns,
- gender disparities,
- urban vs rural differences,
- mobile money adoption trends,
- and financial behavior segmentation.

Visualization techniques included:
- bar charts,
- heatmaps,
- correlation analysis,
- distribution analysis,
- and comparative demographic visualizations.

---

## 3. Feature Engineering

Features used in modeling include:

- Gender
- Age Category
- Urbanization Level
- Country
- Financial Indicator Type
- Time Period

Target Variable:
- Financial exclusion intensity (`OBS_VALUE`)

---

## 4. Machine Learning Modeling

Implemented and compared multiple machine learning models:

### Models Used
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 5. Model Explainability

Used feature importance analysis and explainability techniques to identify:
- the strongest drivers of financial exclusion,
- demographic risk patterns,
- and the impact of digital financial adoption.

---

# Key Insights

Key findings from the analysis include:

- Income constraints remain one of the strongest barriers to financial inclusion.
- Rural populations exhibit significantly higher exclusion rates than urban populations.
- Mobile money adoption is strongly associated with improved financial access.
- Younger populations demonstrate higher engagement with digital financial services.
- Gender disparities continue to influence banking access across several regions.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP

---

# Repository Structure

```bash
modeling-financial-exclusion-africa/
│
├── data/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_model_explainability.ipynb
│
├── visuals/
├── README.md
├── requirements.txt
└── presentation.pdf
