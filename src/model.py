import pandas as pd
import numpy as np
import joblib
import logging
import optuna
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

class ChurnClassifier:
    """
    High-performance churn prediction model with hyperparameter optimization.
    Implements XGBoost with Optuna for automated tuning.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.best_params = None
        self.study = None
        
    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Bayesian optimization of hyperparameters using Optuna.
        
        Returns:
            dict: Optimized hyperparameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'eval_metric': 'aucpr',
                'use_label_encoder': False,
                'tree_method': 'gpu_hist' if self.config["modeling"]["use_gpu"] else 'auto'
            }
            
            model = XGBClassifier(**params)
            
            # Stratified cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True)
            scores = cross_val_score(
                model, X, y, 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            return np.mean(scores)
        
        # Optimization study
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials)
        
        self.best_params = self.study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        return self.best_params
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None
    ) -> XGBClassifier:
        """
        Train final model with optimized parameters.
        Implements early stopping and GPU acceleration.
        """
        try:
            # Model configuration
            params = self.best_params if self.best_params else self.config["modeling"]["xgb_params"]
            params.update({
                'eval_metric': 'aucpr',
                'use_label_encoder': False,
                'tree_method': 'gpu_hist' if self.config["modeling"]["use_gpu"] else 'auto'
            })
            
            self.model = XGBClassifier(**params)
            
            # Early stopping
            eval_set = [(X_train, y_train)]
            if X_test is not None and y_test is not None:
                eval_set.append((X_test, y_test))
                
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=10
            )
            
            # Save model
            model_path = self.config["paths"]["model_output"]
            joblib.dump(self.model, model_path)
            logger.info(f"Saved trained model to {model_path}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation with business metrics.
        """
        try:
            y_pred_proba = self.model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'roc_auc': roc_auc_score(y, y_pred_proba),
                'pr_auc': average_precision_score(y, y_pred_proba),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'roc_curve': roc_curve(y, y_pred_proba),
                'pr_curve': precision_recall_curve(y, y_pred_proba)
            }
            
            logger.info("Model evaluation metrics:")
            for k, v in metrics.items():
                if k not in ['confusion_matrix', 'roc_curve', 'pr_curve']:
                    logger.info(f"{k}: {v:.4f}")
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
