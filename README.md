# 🏡 House Price Prediction App using XGBoost

This project is a **machine learning regression pipeline** to predict California house prices using the **California Housing Dataset**. It includes complete preprocessing, model tuning, evaluation, and a deployed **Streamlit web app** for live predictions.

---

## 🚀 Features

- 📊 EDA + Feature Engineering
- 🧹 Full preprocessing pipeline using `ColumnTransformer`
- ⚙️ Model training with:
  - XGBoost Regressor ✅
  - Random Forest Regressor ✅
- 🔎 Hyperparameter tuning using `GridSearchCV`
- 📈 Evaluation with RMSE & R² score
- 🌐 Live Streamlit app with user inputs

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib (optional)
- Streamlit 🧼

---

## 📂 Folder Structure

```
├── app.py                      # Streamlit app
├── housing.csv                # Dataset
├── best_xgboost_model.pkl     # Trained XGBoost model
├── preprocessing_pipeline.pkl # Feature pipeline
├── main.py                    # Model training and tuning script
├── README.md                  # Project description
```

---

## 📈 How It Works

1. **Preprocess** the dataset (scale, impute, encode)
2. **Train** XGBoost and Random Forest models
3. **Tune** with GridSearchCV for best hyperparameters
4. **Evaluate** using RMSE and R²
5. **Save** the model and preprocessing pipeline
6. **Deploy** a user-friendly web app with Streamlit

---

## 🧪 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/codewithyasho/california-house-price-predictor.git
cd california-house-price-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📷 Screenshot
![app-screenshot](https://via.placeholder.com/800x400.png?text=Streamlit+App+Screenshot)
<img width="1494" height="949" alt="image" src="https://github.com/user-attachments/assets/bd1f0546-e35a-4720-9fb8-de372ecb3cdf" />

---

## 🙋‍♂️ Author

Made with ❤️ by Yashodeep (codewithyasho)

