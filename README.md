# Crop Yield Prediction (Machine Learning Project) – README

Predict crop yields at the country level using machine learning. This project walks through data loading, preprocessing, exploratory data analysis (EDA), model training/evaluation, and deploying a simple web UI for predictions. It uses scikit-learn pipelines with ColumnTransformer to handle both categorical encoding and numeric scaling in one place, ensuring consistent preprocessing for training and inference.

## Features

- End-to-end ML workflow in a Jupyter Notebook
- Clean preprocessing:
  - Remove duplicates and invalid numeric strings
  - One-Hot Encoding for categorical features (Area, Item)
  - Standardization for numeric features (e.g., Year, Average Rainfall, Pesticides, Average Temperature)
- Visual EDA (distribution of countries, crop frequency, total yield per country/crop)
- Multiple regression models trained and compared:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Decision Tree Regressor
  - K-Nearest Neighbors Regressor
- Metrics: Mean Absolute Error (MAE), R² score
- Saved inference pipeline for deployment
- Simple web UI integration to make predictions

## Dataset

- Source: Kaggle – “Crop Yield Prediction” dataset
- Core columns (typical):
  - Year (numeric)
  - Average Rainfall (per year)
  - Pesticides (usage)
  - Average Temperature
  - Area (country)
  - Item (crop name, e.g., Maize, Rice, Wheat, Potato, Soybean)
  - Yield/Production (target)

Note: Column names may vary slightly depending on dataset version. Align the code with your CSV headers (e.g., yield_df.csv).

## Project Structure

- notebooks/
  - crop_yield_prediction.ipynb – Main notebook with EDA, preprocessing, training, evaluation, and inference
- src/
  - preprocessing.py – Helpers for cleaning, type fixes, and pipeline creation
  - train.py – Script to train models and persist the best pipeline
  - predict.py – CLI or function for single prediction using the saved pipeline
- app/
  - app.py – Simple web interface (e.g., Streamlit/Flask) for user inputs and prediction
- models/
  - pipeline.pkl – Trained ColumnTransformer + Regressor
- data/
  - yield_df.csv – Dataset file (not committed; place locally)

You can keep everything in a single notebook to follow along, then modularize as shown for production.

## Setup

1. Clone the repository and open a terminal in the project folder.
2. Create and activate a virtual environment:
   - Python 3.9+ recommended
3. Install dependencies:
   - pandas, numpy, scikit-learn, seaborn, matplotlib
   - streamlit or flask (if deploying a UI)
   - joblib (for model persistence)

Example:
- pip install pandas numpy scikit-learn seaborn matplotlib joblib streamlit

4. Download the dataset and place yield_df.csv in the data/ directory.

## How It Works

1. Load and inspect data:
   - df.shape, df.info(), df.isna().sum(), df.describe()
2. Clean data:
   - Drop duplicates
   - Identify numeric-like strings and convert or drop invalid rows
3. EDA:
   - Country frequency (Area)
   - Crop frequency (Item)
   - Aggregate yield/production per Area and per Item
4. Train/Test Split:
   - 80% train, 20% test
5. Preprocessing with ColumnTransformer:
   - Categorical (Area, Item): OneHotEncoder(handle_unknown="ignore")
   - Numeric (Year, Average Rainfall, Pesticides, Average Temperature): StandardScaler()
   - remainder="passthrough" for any untouched columns (optional)
6. Models:
   - Fit multiple regressors
   - Evaluate on test set with MAE and R²
   - Select best model (low MAE, high R²)
7. Persist:
   - Save full pipeline (preprocessor + best model) via joblib
8. Inference:
   - Wrap a function that accepts raw inputs, applies the same pipeline, and returns prediction
9. UI:
   - Build a simple web form to enter Year, Average Rainfall, Pesticides, Average Temperature, Area, Item, and display predicted yield/production

## Usage

Training from notebook:
- Open notebooks/crop_yield_prediction.ipynb
- Run cells sequentially to:
  - Explore data
  - Build ColumnTransformer
  - Train/evaluate models
  - Save the best pipeline to models/pipeline.pkl

Training via script:
- python src/train.py --data data/yield_df.csv --out models/pipeline.pkl

Single prediction (example code):
- python src/predict.py --pipeline models/pipeline.pkl --year 2013 --rainfall 145.0 --pesticides 121.0 --temperature 16.37 --area "Albania" --item "Maize"

Expected output:
- Predicted production/yield value for the given inputs

Run the web app (Streamlit example):
- streamlit run app/app.py

Then open the local URL provided by Streamlit.

## Example Inference Snippet (Python)

```python
import joblib
import numpy as np

# Load saved pipeline (preprocessor + model)
pipe = joblib.load("models/pipeline.pkl")

# Example inputs
year = 2013
avg_rainfall = 145.0
pesticides = 121.0
avg_temp = 16.37
area = "Albania"
item = "Maize"

X = [[year, avg_rainfall, pesticides, avg_temp, area, item]]
pred = pipe.predict(X)[0]
print(f"Predicted production/yield: {pred:.2f}")
```

## Tips and Gotchas

- Always fit the ColumnTransformer on training data only, then transform both train and test (Pipeline/ColumnTransformer handles this when fit on train and predict on test/new data).
- Use OneHotEncoder(handle_unknown="ignore") to avoid errors with unseen countries/crops at inference time.
- Keep preprocessing and model together in a single Pipeline to prevent training–serving skew.
- Evaluate multiple models; tree-based and KNN often perform competitively on tabular data, but try regularized linear models as baselines.
- Verify units and scales for features (e.g., rainfall per year, pesticides usage, temperature in °C).

## Extending the Project

- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Cross-validation for more robust estimates
- Add weather/time-series features (e.g., seasonal averages, rolling stats)
- Feature importance or SHAP for interpretability
- Persist model versioning and experiment tracking (MLflow/Weights & Biases)
- Containerize app (Docker) and deploy to a cloud platform

## License

This project is for educational purposes. Check dataset licensing on Kaggle before redistribution. Adapt and cite as needed.

## Acknowledgments

- Dataset: Kaggle “Crop Yield Prediction”
- Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib
- Inspired by educational content demonstrating end-to-end ML workflows for agriculture analytics.
