import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReadmissionModelTrainer:
    def __init__(self, data_path='data/processed'):
        self.data_path = Path(data_path)
        self.models_path = Path('models')
        self.models_path.mkdir(exist_ok=True)
        self.results_path = Path('results')
        self.results_path.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

         # Initialize models with preprocessing pipelines
        self.models = {
            'logistic': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'random_forest': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
            ]),
            'xgboost': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
            ])
        }
        
    def load_data(self):
        """Load and prepare preprocessed data"""
        logger.info("Loading preprocessed data...")
        
        # Load data
        X = pd.read_csv(self.data_path / 'X_preprocessed.csv')
        y = pd.read_csv(self.data_path / 'y_preprocessed.csv')
        
        # Log missing value information
        missing_values = X.isnull().sum()
        if missing_values.any():
            logger.info("Missing values found in features:")
            for col, count in missing_values[missing_values > 0].items():
                logger.info(f"{col}: {count} missing values")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}")
        return self
    
    def train_models(self):
        """Train all models"""
        logger.info("Training models...")
        
        self.trained_models = {}
        self.predictions = {}
        self.scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train.values.ravel())
            
            # Generate predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Store results
            self.trained_models[name] = model
            self.predictions[name] = {
                'pred': y_pred,
                'prob': y_pred_proba
            }
            
            # Calculate metrics
            self.scores[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
            }
            
        return self
    
    def create_evaluation_visualizations(self):
        """Generate evaluation visualizations"""
        logger.info("Creating evaluation visualizations...")
        
        # 1. ROC Curves
        plt.figure(figsize=(10, 6))
        for name, preds in self.predictions.items():
            fpr, tpr, _ = roc_curve(self.y_test, preds['prob'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {self.scores[name]["roc_auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(self.results_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for (name, preds), ax in zip(self.predictions.items(), axes):
            cm = confusion_matrix(self.y_test, preds['pred'])
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_title(f'{name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance (for Random Forest)
        # Access the classifier from inside the pipeline
        rf_classifier = self.trained_models['random_forest'].named_steps['classifier']
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': rf_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Most Important Features (Random Forest)')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig(self.results_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self
    
    def save_results(self):
        """Save model results and metrics"""
        logger.info("Saving results...")
        
        # Save metrics
        results_df = pd.DataFrame(self.scores).T
        results_df.to_csv(self.results_path / 'model_metrics.csv')
        
        # Save models
        for name, model in self.trained_models.items():
            model_path = self.models_path / f'{name}_model.pkl'
            pd.to_pickle(model, model_path)
        
        # Create summary report
        with open(self.results_path / 'summary_report.txt', 'w') as f:
            f.write("NHS Readmission Prediction - Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            for name, scores in self.scores.items():
                f.write(f"\n{name.upper()} MODEL:\n")
                f.write("-" * 20 + "\n")
                for metric, value in scores.items():
                    f.write(f"{metric}: {value:.3f}\n")
            
        return self
    
    def run_training_pipeline(self):
        """Run complete training and evaluation pipeline"""
        try:
            (self.load_data()
                 .train_models()
                 .create_evaluation_visualizations()
                 .save_results())
            
            logger.info("Training pipeline completed successfully!")
            
            # Print summary
            print("\nModel Performance Summary:")
            print("=" * 50)
            for name, scores in self.scores.items():
                print(f"\n{name.upper()}:")
                for metric, value in scores.items():
                    print(f"{metric}: {value:.3f}")
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = ReadmissionModelTrainer()
    trainer.run_training_pipeline()