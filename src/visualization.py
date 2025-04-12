import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging
import joblib

logger = logging.getLogger(__name__)

class ChurnVisualizer:
    """
    Advanced visualization toolkit for churn prediction analysis.
    Generates publication-quality plots and interactive visualizations.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.style = config["visualization"]["style"]
        self.palette = config["visualization"]["palette"]
        sns.set_style(self.style)
        sns.set_palette(self.palette)
        
    def plot_feature_importance(
        self, 
        model: Any, 
        feature_names: list,
        top_n: int = 20,
        figsize: tuple = (12, 8)
    ) -> plt.Figure:
        """
        Plot feature importance with confidence intervals.
        """
        try:
            # Get feature importance
            importance = model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            indices = np.argsort(importance)[-top_n:][::-1]
            
            # Create plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_title("Feature Importance with Standard Deviation")
            
            # Plot bars
            bars = ax.barh(
                range(top_n), 
                importance[indices][:top_n][::-1], 
                xerr=std[indices][:top_n][::-1],
                align='center'
            )
            
            # Customize
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(feature_names[indices][:top_n][::-1])
            ax.set_xlabel("Mean Importance Score")
            ax.set_ylabel("Features")
            plt.tight_layout()
            
            # Save figure
            fig_path = f"{self.config['paths']['reports']}/figures/feature_importance.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {fig_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            raise
    
    def plot_roc_curve(
        self, 
        fpr: np.ndarray, 
        tpr: np.ndarray, 
        roc_auc: float,
        figsize: tuple = (8, 6)
    ) -> plt.Figure:
        """Plot ROC curve with AUC score."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        
        fig_path = f"{self.config['paths']['reports']}/figures/roc_curve.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        return fig
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray,
        classes: list = ['Not Churn', 'Churn'],
        figsize: tuple = (8, 6)
    ) -> plt.Figure:
        """Plot annotated confusion matrix."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        fig_path = f"{self.config['paths']['reports']}/figures/confusion_matrix.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        return fig
    
    def generate_all_visualizations(
        self, 
        model: Any, 
        feature_names: list,
        metrics: Dict[str, float]
    ):
        """Generate complete visualization suite."""
        self.plot_feature_importance(model, feature_names)
        self.plot_roc_curve(
            metrics['roc_curve'][0], 
            metrics['roc_curve'][1], 
            metrics['roc_auc']
        )
        self.plot_confusion_matrix(np.array(metrics['confusion_matrix']))

