# 🌲🔥 Algerian Forest Fire Predictor

A Machine Learning web application built with **Flask** that predicts **Fire Weather Index (FWI)** and **forest fire occurrence (Fire / No Fire)** for Algerian forests using meteorological and vegetation indices. The app uses pre-trained regression and classification models to provide real-time fire risk insights.

---

## 📘 Overview

This project uses the **Algerian Forest Fires Dataset** to model and predict:  
- **Fire Weather Index (FWI)** using regularized linear regression models.  
- **Fire Occurrence (Fire / No Fire)** using logistic regression classifiers.  

Users provide meteorological inputs through a web interface, and the application returns both the predicted FWI value and the likelihood of a forest fire.

---

## 🚀 Features

### 🔥 Fire Risk Prediction (Classification)

Predicts whether there is a **risk of forest fire** using Logistic Regression-based models.

- GridSearchCV-optimized Logistic Regression  
- RandomizedSearchCV-optimized Logistic Regression  

These models classify the input as **Fire** or **No Fire**, based on learned decision boundaries from the dataset.

### 📈 FWI Prediction (Regression)

Predicts the **Fire Weather Index (FWI)**, an indicator of potential fire intensity.

Models used with cross-validated hyperparameter tuning:

- **Ridge Regression** (L2 regularization)
- **Lasso Regression** (L1 regularization)
- **ElasticNet Regression** (combined L1 + L2)

Each model is trained on scaled features and saved as a `.pkl` file for fast inference.

### 🖥️ Web Application

- Built using **Flask**
- User-friendly **HTML + CSS** interface
- Takes meteorological inputs from an HTML form
- Uses serialized `.pkl` models for live predictions
- Displays:
  - Fire risk: **Fire / No Fire**
  - Predicted **FWI** value

---

## 🧠 Machine Learning Models

### 🔷 FWI Regression Models

| Model              | Technique          | Notes                     |
|--------------------|-------------------|---------------------------|
| Ridge Regression   | L2 Regularization | Reduces overfitting       |
| Lasso Regression   | L1 Regularization | Performs feature selection|
| ElasticNet         | L1 + L2           | Balanced regularization   |

- All models trained with cross-validation.
- Hyperparameters tuned using appropriate CV strategies.
- Numerical features scaled using a fitted `scaler.pkl`.

### 🔶 Fire Classification Models

| Model                                | Search Method       | Purpose                            |
|--------------------------------------|---------------------|------------------------------------|
| Logistic Regression (GridSearchCV)   | Exhaustive search   | High-accuracy parameter selection  |
| Logistic Regression (RandomSearchCV) | Randomized search   | Faster, scalable tuning            |

Both classifiers are saved inside `models/class-predictors/` as `.pkl` files and loaded by the Flask app at runtime.

---

## 📊 Dataset

**Dataset:** Algerian Forest Fires Dataset (UCI / Kaggle variant)

The dataset contains **meteorological** and **vegetation** indices used for both classification (fire / no fire) and regression (FWI) tasks. It includes observations from Algerian regions over a fire season, with each instance labeled accordingly.

**Key Features:**

- Temperature  
- Relative Humidity (RH)  
- Wind Speed  
- Rain  
- FFMC (Fine Fuel Moisture Code)  
- DMC (Duff Moisture Code)  
- DC (Drought Code)  
- ISI (Initial Spread Index)  
- BUI (Buildup Index)  
- FWI (Fire Weather Index)  
- Fire Label (fire / no fire)  

**Steps Performed:**

- Data preprocessing and cleaning  
- Handling missing values and inconsistent categories  
- Feature engineering on meteorological and index-based attributes  
- Scaling numerical features (for regression models)  
- Exploratory Data Analysis (correlation heatmaps, distributions, feature importance)  
- Model training and evaluation (regression and classification)  
- Hyperparameter tuning with **GridSearchCV** and **RandomizedSearchCV**  
- Saving final models as `.pkl` files

---

```
## 📁 Project Structure

Forest-Fire-Predictor/
│
├── datasets/
│   ├── algerian-forest-fires-cleaned.csv
│   └── algerian-forest-fires.csv
│
├── models/
│   ├── class-predictors/
│   │   ├── gridsearch.pkl
│   │   └── randomsearch.pkl
│   │
│   └── fwi-predictors/
│       ├── elasticnet.pkl
│       ├── lasso.pkl
│       ├── ridge.pkl
│       └── scaler.pkl
│
├── static/
│   ├── homepage_style.css
│   └── input_style.css
│
├── templates/
│   ├── homepage.html
│   └── userinput.html
│
├── 1_preprocessing_eda_fe.ipynb
├── 2_training_linear_model_fwi.ipynb
├── 3_training_logistic_model_class.ipynb
│
├── application.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

1️⃣ **Clone the repository**

- git clone https://github.com/HaZaRdOuSDeVeLoPeR/Forest-Fire-Predictor.git
- cd Forest-Fire-Predictor

2️⃣ **Install dependencies**

pip install -r requirements.txt


3️⃣ **Run the Flask application**

python application.py


4️⃣ **Open in browser**

Go to: http://127.0.0.1:8080/

---

## 🧪 Usage

1. Open the web app in your browser.  
2. Enter the following meteorological and index parameters:
   - Temperature  
   - Relative Humidity  
   - Wind Speed  
   - Rain  
   - FFMC, DMC, DC, ISI, BUI  
3. Select:
   - **FWI Regression Model**: Ridge / Lasso / ElasticNet  
   - **Fire Classification Model**: GridSearch / RandomSearch  
4. Click **Predict** to get:
   - Fire Prediction: **Fire** / **No Fire**  
   - Predicted **FWI** value  

---

## 🎨 Tech Stack

- **Python**
- **Flask**
- **HTML + CSS**
- **Scikit-Learn**
- **Pandas**, **NumPy**
- **Pickle** (for model serialization)
- **Jupyter Notebooks** (for EDA, feature engineering, and model training)

---

## 📦 Model Files

All trained models are stored inside the `models/` directory.

### 🚩 Classification Models

Location: `models/class-predictors/`

- `gridsearch.pkl` – Logistic Regression model tuned via GridSearchCV  
- `randomsearch.pkl` – Logistic Regression model tuned via RandomizedSearchCV  

### 📈 Regression Models

Location: `models/fwi-predictors/`

- `ridge.pkl` – Ridge Regression model  
- `lasso.pkl` – Lasso Regression model  
- `elasticnet.pkl` – ElasticNet Regression model  
- `scaler.pkl` – Fitted scaler for input normalization  

---

## 📝 Future Improvements

- Add an interactive visualization dashboard (e.g., FWI vs. key parameters)  
- Deploy on **Render / Heroku / AWS** for public access  
- Integrate **real-time weather APIs** for live predictions  
- Add **SHAP-based explanations** for model interpretability  

---

## 👨‍💻 Author

**Aditya Vimal**  
B.Tech CSE — NIT Warangal
