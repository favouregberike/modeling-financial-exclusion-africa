# Modeling Financial Exclusion Drivers in Africa

## Project Overview

Financial inclusion remains one of the most important drivers of economic participation and social development across Africa. Despite the rapid growth of digital finance and mobile money services, millions of people remain excluded from formal financial systems.

This project analyzes financial exclusion patterns across African populations using the World Bank Global Findex dataset. The goal is to identify demographic and structural factors associated with financial exclusion and understand the barriers preventing access to financial services.

Using exploratory data analysis and machine learning techniques, this project models financial exclusion indicators across countries, age groups, gender categories, and urbanization levels to generate actionable insights for policymakers, fintech companies, and development organizations.

---

## Objectives

The objectives of this project are to:

- Analyze demographic patterns of financial exclusion
- Identify the strongest barriers to banking access
- Examine trends in digital financial adoption
- Model financial exclusion indicators using machine learning
- Generate policy and business recommendations

---

## Dataset

Source:
World Bank Global Findex Database

Dataset contains:
- Financial inclusion indicators
- Banking access statistics
- Mobile money usage
- Remittance activity
- Savings and borrowing behavior
- Demographic segmentation
- Country-level observations

Key dimensions include:
- Country
- Gender
- Age group
- Urbanization
- Financial activity indicators
- Time period

---

## Problem Statement

Millions of adults across Africa remain unbanked due to factors such as:
- insufficient income,
- lack of trust in financial institutions,
- distance from banking services,
- and limited documentation.

Understanding the drivers of financial exclusion is critical for:
- governments,
- financial institutions,
- fintech companies,
- and development organizations seeking to expand financial access.

This project aims to uncover the structural and demographic factors most associated with financial exclusion across African populations.

---

## Project Workflow

### 1. Data Cleaning
- Handling missing values
- Filtering relevant indicators
- Encoding categorical variables
- Preparing analytical datasets

### 2. Exploratory Data Analysis (EDA)
- Country-level financial exclusion analysis
- Gender-based comparisons
- Age-group analysis
- Urban vs rural financial access
- Mobile money adoption trends

### 3. Feature Engineering
Features used include:
- Gender
- Age category
- Urbanization level
- Country
- Indicator category
- Time period

### 4. Machine Learning
Models explored:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### 5. Model Evaluation
Evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 6. Insight Generation
- Identification of major exclusion drivers
- Digital finance adoption patterns
- Demographic risk segmentation
- Policy recommendations

---

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP

---

## Key Insights

Some of the major insights explored in this project include:

- Income constraints are among the strongest barriers to account ownership
- Rural populations show higher exclusion rates compared to urban populations
- Mobile money adoption is strongly associated with improved financial inclusion
- Younger populations rely more heavily on digital financial channels
- Gender disparities continue to affect access to financial services in several regions

---

## Repository Structure

```bash
modeling-financial-exclusion-africa/
│
├── data/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_storytelling.ipynb
│
├── visuals/
├── README.md
├── requirements.txt
└── presentation.pdf
