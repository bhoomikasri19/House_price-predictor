
# 🏠 House Price Predictor

A machine learning web app that predicts house prices in Mumbai, India (in Lakhs) based on property features like carpet area, floor, bathrooms, balconies, and location.

---

## 🎯 Features
- Predicts house price in Lakhs based on user inputs
- Supports multiple Mumbai locations from real housing data
- Interactive web interface built with Streamlit

---

## 🧠 Model Details
- **Algorithm:** Random Forest Regressor (10 trees)
- **Dataset:** House Prices (Kaggle) — 187,000+ real listings
- **R² Score:** 0.85
- **Target:** Price in Lakhs and Cores

---

## 🛠️ Data Preprocessing
- Removed irrelevant columns (Title, Description, Society, etc.)
- Extracted numeric values from Carpet Area and Floor strings
- Converted prices from mixed format 
- Handled missing values using mean/median imputation
- Removed outliers using 1st and 99th percentile on price per sqft
- One-hot encoded Location column

---

## 💻 Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (Random Forest, StandardScaler, Pipeline)
- Streamlit, Joblib

---

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/bhoomikasri19/House-price-predictor.git
cd House-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app (model is pre-trained)
streamlit run app.py
```

> **Note:** The dataset (187k rows, 31MB) is not included due to size.
> Download from Kaggle: https://www.kaggle.com/datasets/juhibhojani/house-price
> Place `house_prices.csv` in root folder and run `python test.py` to retrain.

---

## 📁 Project Structure

```
House-price-predictor/
├── app.py                  # Streamlit web app
├── test.py                 # Model training script
├── requirements.txt        # Dependencies
├── README.md
└── model/
    ├── houseprice.pkl      # Trained Random Forest model
    └── columns.pkl         # Feature columns
```

---

## 📊 Model Comparison

| Model | R² Score |
|---|---|
| Linear Regression | 0.63 |
| Random Forest (100 trees) | 0.86 |
| Random Forest (10 trees) | 0.85 ✅ deployed |

---

