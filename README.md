# Real Estate Sale Price and Sales Ratio Prediction

## Overview
This project analyzes Conneticut real estate data to predict two targets: `Sale Amount` and `Sales Ratio` (`Sale Amount` / `Assessed Value`). The goal is to find which features and models best predict each target and to understand their predictive reliability.

## Data
Data sourced from [Connecticut Open Data](https://data.ct.gov/Housing-and-Development/Real-Estate-Sales-2001-2023-GL/5mzw-sjtu/about_data)

The dataset contains property listings with features including:  
- `Assessed Value`  
- `Property Type`, `Town`, `Street`  
- `Year Sold`, `List Year`, `Years on Market`  
- `Assessor Remarks`, `OPM Remarks`  
- Geographic coordinates (`X Location`, `Y Location`)  
- Temporal features (sin/cos transformations of day of year and day of week sold)

Targets:  
- **Sale Amount** – The actual sale price of the property  
- **Sales Ratio** – `Sale Amount` / `Assessed Value`  

---

## Models
### Linear Models
- **Ridge Regression**  
- **Linear Regression**  
- **ARD Regression**

These models were primarily used for predicting **Sale Amount**, as it has a strong linear relationship with `Assessed Value`.

### Tree-Based Models
- **XGBoost**  
- **LightGBM**  

These models were primarily used for predicting **Sales Ratio**, which exhibits non-linear relationships influenced by market conditions (`List Year`, historic events).  

Hyperparameters for all models were tuned using `GridSearchCV`.

---

## Results

### Sale Amount
- Ridge Regression achieved $R^2 \approx 0.9$
- `Assessed Value` alone explains most of the variance ($R^2 \approx 0.9$)  
- Removing `Assessed Value` drops performance drastically ($R^2 \approx 0.18$)  

**Insight:** Sale Amount can be reliably predicted from Assessed Value, allowing buyers and sellers to make informed pricing decisions.

### Sales Ratio
- LIGHTGBM achieved $R^2 \approx 0.51$
- Feature importance analysis shows `List Year` dominates
- Partial dependence plots reveal spikes around 2008 (housing crash) and dips around 2020 (COVID-19 pandemic)  

**Insight:** Sales Ratio is influenced by features outside of the dataset, making precise prediction difficult. For listings since 2023, performance drops to ($R^2 \approx 0.24$).

---

## Usage

### Install dependencies
```bash
pip install -r requirements.txt