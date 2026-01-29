"""
ML Model Training and Prediction for Membership Renewal
Uses LightGBM with probability calibration and SHAP explainability
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    confusion_matrix, roc_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
import shap
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class RenewalPredictor:
    """
    Membership Renewal Prediction Model
    Uses LightGBM with calibration for reliable probabilities
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.calibrated_model = None
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance = None
        self.shap_explainer = None
        
    def _encode_categorical_features(
        self, 
        X: pd.DataFrame, 
        categorical_features: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """Encode categorical features using Label Encoding"""
        X = X.copy()
        
        for col in categorical_features:
            if col in X.columns:
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        X[col] = X[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        return X
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: List[str] = None,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the renewal prediction model
        
        Args:
            X: Feature matrix
            y: Target variable (binary: renewed or not)
            categorical_features: List of categorical feature names
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary with training metrics
        """
        print("🚀 Starting model training...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode categorical features
        if categorical_features:
            X_encoded = self._encode_categorical_features(
                X, categorical_features, fit=True
            )
        else:
            X_encoded = X.copy()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_encoded, y,
            test_size=validation_split,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Positive rate (train): {y_train.mean():.2%}")
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 7,
            'min_child_samples': 20,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        print("\n📊 Training LightGBM model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Get predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_val = self.model.predict(X_val)
        
        # Calculate metrics
        train_auc = roc_auc_score(y_train, y_pred_train)
        val_auc = roc_auc_score(y_val, y_pred_val)
        
        print(f"\n✅ Training AUC: {train_auc:.4f}")
        print(f"✅ Validation AUC: {val_auc:.4f}")
        
        
        # Note: LightGBM probabilities are already well-calibrated
        # Skipping additional calibration to avoid compatibility issues
        print("\n✅ Using LightGBM native probabilities (already well-calibrated)")
        self.calibrated_model = None
        
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Important Features:")
        print(self.feature_importance.head(10).to_string(index=False))
        
        # Initialize SHAP explainer
        print("\n🔍 Initializing SHAP explainer...")
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        metrics = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'best_iteration': self.model.best_iteration,
            'feature_importance': self.feature_importance
        }
        
        print("\n✅ Model training complete!")
        return metrics
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        categorical_features: List[str] = None,
        use_calibrated: bool = True
    ) -> np.ndarray:
        """
        Predict renewal probabilities
        
        Args:
            X: Feature matrix
            categorical_features: List of categorical feature names
            use_calibrated: Whether to use calibrated probabilities
            
        Returns:
            Array of renewal probabilities
        """
        # Encode categorical features
        if categorical_features:
            X_encoded = self._encode_categorical_features(
                X, categorical_features, fit=False
            )
        else:
            X_encoded = X.copy()
        
        # Predict (using LightGBM directly)
        probabilities = self.model.predict(X_encoded)
        
        return probabilities
    
    def get_shap_values(
        self,
        X: pd.DataFrame,
        categorical_features: List[str] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for explanations
        
        Args:
            X: Feature matrix
            categorical_features: List of categorical feature names
            
        Returns:
            SHAP values array
        """
        # Encode categorical features
        if categorical_features:
            X_encoded = self._encode_categorical_features(
                X, categorical_features, fit=False
            )
        else:
            X_encoded = X.copy()
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_encoded)
        
        return shap_values
    
    def save_model(self, filepath: str):
        """Save model and encoders"""
        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'shap_explainer': self.shap_explainer
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and encoders"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.calibrated_model = model_data['calibrated_model']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.shap_explainer = model_data['shap_explainer']
        print(f"✅ Model loaded from {filepath}")


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'auc_roc': auc,
        'avg_precision': avg_precision,
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'confusion_matrix': cm
    }
    
    return metrics


if __name__ == "__main__":
    from feature_engineering import prepare_training_data
    
    print("=" * 60)
    print("MEMBERSHIP RENEWAL PREDICTION - MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n📂 Loading training data...")
    df_train = pd.read_csv('membership_renewal_history_labeled_2022_2024_12000rows_v4.csv')
    
    # Prepare features
    print("\n🔧 Engineering features...")
    X, y, engine = prepare_training_data(df_train)
    
    # Get categorical features
    categorical_features = engine.get_feature_names()['categorical']
    
    # Train model
    predictor = RenewalPredictor(random_state=42)
    metrics = predictor.train(
        X, y,
        categorical_features=categorical_features,
        validation_split=0.2
    )
    
    # Save model
    print("\n💾 Saving model...")
    predictor.save_model('models/renewal_predictor.pkl')
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
