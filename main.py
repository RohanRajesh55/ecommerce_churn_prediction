import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model.__class__.__name__} Confusion Matrix")
    plt.show()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {metrics["ROC-AUC"]:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model.__class__.__name__} ROC Curve")
    plt.legend()
    plt.show()
    return metrics

def main():
    file_path = r"C:\ecommerce_churn_prediction\data\processed\preprocessed_data.csv"
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=[np.number])
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    logger.info("Class Distribution After SMOTE: %s", pd.Series(y_train_bal).value_counts())
    
    logistic_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
    logistic_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid=logistic_param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )
    logistic_grid.fit(X_train_bal, y_train_bal)
    best_logistic = logistic_grid.best_estimator_
    
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'class_weight': [None, 'balanced']
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )
    rf_grid.fit(X_train_bal, y_train_bal)
    best_rf = rf_grid.best_estimator_
    
    scale_pos_weight = (len(y_train_bal) - sum(y_train_bal)) / sum(y_train_bal)
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, eval_metric='logloss'),
        param_grid=xgb_param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )
    xgb_grid.fit(X_train_bal, y_train_bal)
    best_xgb = xgb_grid.best_estimator_
    
    logger.info("Evaluating Logistic Regression")
    log_metrics = evaluate_model(best_logistic, X_test, y_test)
    logger.info("Logistic Regression Metrics: %s", log_metrics)
    
    logger.info("Evaluating Random Forest")
    rf_metrics = evaluate_model(best_rf, X_test, y_test)
    logger.info("Random Forest Metrics: %s", rf_metrics)
    
    logger.info("Evaluating XGBoost")
    xgb_metrics = evaluate_model(best_xgb, X_test, y_test)
    logger.info("XGBoost Metrics: %s", xgb_metrics)
    
    best_model = best_xgb  # Choose the best based on performance; adjust as per your evaluation
    model_save_path = r"C:\ecommerce_churn_prediction\models\best_model.pkl"
    joblib.dump(best_model, model_save_path)
    logger.info("Best model saved to %s", model_save_path)
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importance.values, y=feature_importance.index)
        plt.title("Feature Importance")
        plt.show()

if __name__ == '__main__':
    main()