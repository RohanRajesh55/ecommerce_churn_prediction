import pandas as pd
import numpy as np
import joblib
import logging
import optuna
from typing import Dict, Any, Tuple, Optional
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChurnClassifier:
    """
    High-performance churn prediction model with hyperparameter optimization using XGBoost and Optuna.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the ChurnClassifier with the provided configuration.

        Parameters:
            config (dict): Configuration with keys for modeling parameters,
                           paths for model output and study output, and GPU settings.
        """
        self.config = config
        self.model: Optional[XGBClassifier] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.study: Optional[optuna.study.Study] = None

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna's Bayesian optimization.

        Parameters:
            X (pd.DataFrame): Features.
            y (pd.Series): Target labels.
            n_trials (int): Number of trials for optimization.

        Returns:
            Dict[str, Any]: Best hyperparameters found.
        """
        def objective(trial: optuna.trial.Trial) -> float:
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
                'tree_method': 'gpu_hist' if self.config["modeling"].get("use_gpu", False) else 'auto'
            }

            model = XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            return np.mean(scores)

        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials)

        self.best_params = self.study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")

        # Save Optuna study for future analysis
        optuna_study_path: str = self.config["paths"]["optuna_study_output"]
        joblib.dump(self.study, optuna_study_path)
        logger.info(f"Saved Optuna study to {optuna_study_path}")

        return self.best_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> XGBClassifier:
        """
        Train the XGBoost classifier using optimized hyperparameters or default parameters.

        Parameters:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame, optional): Test features for evaluation.
            y_test (pd.Series, optional): Test labels for evaluation.

        Returns:
            XGBClassifier: The trained model.
        """
        try:
            # Use optimized parameters if available; otherwise, use default parameters from config.
            if self.best_params:
                params = self.best_params.copy()
            else:
                params = self.config["modeling"]["xgb_params"].copy()

            # Update parameters with common settings.
            params.update({
                'eval_metric': 'aucpr',
                'use_label_encoder': False,
                'tree_method': 'gpu_hist' if self.config["modeling"].get("use_gpu", False) else 'auto'
            })

            self.model = XGBClassifier(**params)

            # Prepare evaluation set if test data is provided.
            eval_set = [(X_train, y_train)]
            if X_test is not None and y_test is not None:
                eval_set.append((X_test, y_test))

            # Attempt to use early_stopping_rounds.
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=50,
                    verbose=10
                )
            except TypeError as e:
                logger.warning("early_stopping_rounds not supported in this version of XGBClassifier; training without early stopping.")
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=10
                )

            model_path: str = self.config["paths"]["model_output"]
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
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model using performance metrics.

        Parameters:
            X (pd.DataFrame): Feature set for evaluation.
            y (pd.Series): True labels.
            threshold (float): Classification threshold (default 0.5).

        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet.")

            y_pred_proba = self.model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

            metrics: Dict[str, Any] = {
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