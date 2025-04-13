# E-commerce Customer Churn Prediction

This project is an end-to-end machine learning solution that predicts customer churn for e-commerce businesses. It covers **data ingestion**, **preprocessing**, **model training** (with hyperparameter tuning), **evaluation**, and **deployment** through an interactive web application built using Streamlit.

---

## 🚀 Project Overview

The primary goal is to identify customers who are likely to churn, empowering businesses to retain them proactively. The project consists of the following components:

- **Data Ingestion & Preprocessing:**  
  Reads data from an Excel file and processes it using a scikit-learn pipeline. Includes imputation, scaling, and one-hot encoding for numeric and categorical features.

- **Model Training & Tuning:**  
  Trains an XGBoost classifier with optional hyperparameter tuning using Optuna. The model is evaluated using metrics such as accuracy, precision, recall, F1 score, ROC-AUC, and PR-AUC.

- **Deployment:**  
  Saves the trained model as a full pipeline (preprocessor + model). Provides an interactive Streamlit web app (`deploy.py`) for predictions based on user inputs.

---

## 🧱 Project Structure

Here's how the repository is organized:

```
ecommerce_churn_prediction/
├── app/
│   └── deploy.py          # Streamlit deployment script
├── data/
│   └── raw/
│       └── E Commerce Dataset.xlsx  # Raw dataset
├── models/
│   ├── best_model.pkl     # Trained model file
│   └── full_pipeline_model.pkl  # Preprocessor + model pipeline
├── notebooks/
│   └── model_development.ipynb  # EDA & experimentation notebook
├── src/
│   ├── data_loader.py     # Module for loading data
│   ├── model.py           # Module for training & evaluation
│   └── preprocessor.py    # Module for preprocessing pipeline
├── config.yml             # Configuration file for paths & parameters
├── .env                   # Environment variables (if needed)
├── .gitignore             # Git ignore file
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

---

## ⚙️ Installation

Follow these steps to set up the project:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ecommerce_churn_prediction.git
   cd ecommerce_churn_prediction
   ```

2. **Set Up a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate      # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **(Optional) Configure Environment Variables:**

   If the project uses sensitive settings (like API keys), add them in a `.env` file:

   ```
   YOUR_VARIABLE=value
   ```

---

## 🔧 Configuration

Project settings (e.g., file paths, model parameters) are managed in the `config.yml` file. Here's an example configuration:

```yaml
paths:
  raw_data: "data/raw/E Commerce Dataset.xlsx"
  model_output: "models/best_model.pkl"
  model_output_full: "models/full_pipeline_model.pkl"

data:
  sheet_name: "E Comm"
  target: "Churn"
  test_size: 0.2

modeling:
  use_gpu: false
  xgb_params:
    n_estimators: 300
    max_depth: 5
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
```

---

## 🧪 How to Run

### 1. Train and Evaluate the Model:

Run the following to execute the full training pipeline:

```bash
python main.py
```

What this does:

- Loads and preprocesses the dataset.
- Trains the XGBoost model with optional hyperparameter tuning.
- Evaluates performance metrics.
- Saves the standalone model and the preprocessor + model pipeline.

### 2. Deploy the Model with Streamlit:

Start the interactive web app with:

```bash
streamlit run app/deploy.py
```

You’ll see a form in your browser where you can input customer details to get churn predictions and probabilities.

---

## 🧩 Dependencies

Main dependencies include:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `optuna`
- `joblib`
- `pyyaml`
- `streamlit`

Refer to `requirements.txt` for the full list.

---

## 🌟 Future Enhancements

Here’s what’s planned next:

- ✅ Add unit tests for data loading and preprocessing
- ✅ Implement model monitoring in production
- ✅ Enhance the Streamlit UI
- ✅ Improve overall documentation

---
