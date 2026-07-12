# 🧠 ML Arena – Intelligent Machine Learning Benchmark Platform

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-green)
![Status](https://img.shields.io/badge/Development-Active-orange)

---

# 🚀 Overview

**ML Arena** is an interactive machine learning platform that simplifies the process of building and evaluating machine learning models through an easy-to-use Streamlit interface.

The platform provides a structured workflow where users can upload datasets, understand data characteristics, preprocess data, train machine learning models, and analyze model performance using evaluation metrics.

The main goal of ML Arena is to create a simple yet effective environment for experimenting with different machine learning algorithms without manually writing repetitive ML pipeline code.

---

# ✨ Implemented Features

## 📂 Dataset Upload Module

* Upload datasets in CSV format
* Store uploaded data for further processing
* Display dataset preview
* Validate dataset availability before processing

---

## 📊 Dataset Analysis Module

Provides automatic dataset insights:

* Dataset shape information
* Column details
* Data type analysis
* Missing value detection
* Statistical summary

---

## ⚙️ Data Preprocessing Module

Implemented preprocessing workflow:

* Feature and target selection
* Data preparation before model training
* Dataset transformation pipeline

---

# 🧠 Machine Learning Model Training

ML Arena currently supports supervised learning workflows.

## Regression

Implemented regression model training:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

## Classification

Implemented classification model training:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

---

# 📈 Model Evaluation

## Regression Evaluation

The platform calculates:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## Classification Evaluation

The platform generates:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Classification Report

---

# 🖥️ Results Dashboard

Implemented results visualization page that displays:

* Selected model information
* Prediction results
* Evaluation metrics
* Classification performance report

---

# 🏗️ Application Workflow

```
Dataset Upload
        |
        ↓
Dataset Analysis
        |
        ↓
Data Preprocessing
        |
        ↓
Model Training
        |
        ↓
Model Evaluation
        |
        ↓
Results Display
```

---

# 🛠️ Technology Stack

## Programming Language

* Python

## Framework

* Streamlit

## Data Processing

* Pandas
* NumPy

## Machine Learning

* Scikit-learn

## Visualization

* Matplotlib
* Plotly
* Seaborn

## Model Saving / Loading

* Joblib

---

# 📁 Project Structure

```
ML-Arena/

│
├── app.py
│
├── pages/
│   ├── Dashboard.py
│   ├── Upload_Dataset.py
│   ├── Dataset_Analysis.py
│   ├── Preprocessing.py
│   ├── Model_Training.py
│   └── Results.py
│
├── utils/
│   ├── preprocessing.py
│   ├── model_utils.py
│   └── theme.py
│
├── assets/
│   └── css/
│       └── style.css
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation & Setup

Clone the repository:

```bash
git clone <repository-url>
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 📌 Current Project Status

Implemented Modules:

✅ Streamlit Application Setup
✅ Multi-page Architecture
✅ Dataset Upload
✅ Dataset Analysis
✅ Data Preprocessing Workflow
✅ Regression Model Training
✅ Classification Model Training
✅ Model Performance Evaluation
✅ Results Dashboard
✅ Custom UI Styling

---

# 🔮 Future Improvements

Future versions may include additional machine learning automation and advanced analysis capabilities.

---

# 👩‍💻 Developer

**Hanshika Mukati**

B.Tech Computer Science Engineering

Machine Learning | Python | Artificial Intelligence

---

## 📍 Project Status

🚧 Active Development

Current Version: **v0.3**

```
```
