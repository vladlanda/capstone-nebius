import argparse
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, roc_auc_score, classification_report)
from sklearn.model_selection import cross_val_score
import wandb
from config import config


class MixtureOfExpertsRegressor(BaseEstimator, RegressorMixin):
    """
    Mixture-of-Experts model for regression:
    1. Classifier predicts high vs low rating
    2. Two specialized regressors for each segment
    3. Combines predictions using soft or hard gating
    """
    
    def __init__(self, classifier, regressor_low, regressor_high, 
                 threshold=90, use_soft_gate=True, target_min=0, target_max=5):
        self.classifier = classifier
        self.regressor_low = regressor_low
        self.regressor_high = regressor_high
        self.threshold = threshold
        self.use_soft_gate = use_soft_gate
        self.target_min = target_min
        self.target_max = target_max
        
    def fit(self, X, y):
        """Train the mixture of experts model."""
        # Handle both DataFrame and array inputs
        if isinstance(y, pd.DataFrame):
            y_values = y.values.ravel()
        elif isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y.ravel() if len(y.shape) > 1 else y
        
        # Create binary labels for classification
        y_binary = (y_values >= self.threshold).astype(int)
        
        # Train classifier on full dataset
        print(f"\n  Training classifier (threshold={self.threshold})...")
        print(f"    Low class: {(y_binary == 0).sum()} samples")
        print(f"    High class: {(y_binary == 1).sum()} samples")
        self.classifier.fit(X, y_binary)
        
        # Split data based on true labels
        mask_low = y_binary == 0
        mask_high = y_binary == 1
        
        X_low, y_low = X[mask_low], y_values[mask_low]
        X_high, y_high = X[mask_high], y_values[mask_high]
        
        # Train specialized regressors
        print(f"\n  Training low-rating regressor on {len(X_low)} samples...")
        self.regressor_low.fit(X_low, y_low)
        
        print(f"  Training high-rating regressor on {len(X_high)} samples...")
        self.regressor_high.fit(X_high, y_high)
        
        return self
    
    def predict(self, X):
        """Predict using mixture of experts with soft or hard gating."""
        # Get classification probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probs = self.classifier.predict_proba(X)
            p_high = probs[:, 1]  # Probability of high class
        else:
            # Fallback for classifiers without predict_proba
            p_high = self.classifier.predict(X).astype(float)
        
        # Get predictions from both regressors
        y_low = self.regressor_low.predict(X).ravel()
        y_high = self.regressor_high.predict(X).ravel()
        
        if self.use_soft_gate:
            # Soft gating: weighted average based on probabilities
            predictions = (1 - p_high) * y_low + p_high * y_high
        else:
            # Hard gating: use classifier decision
            predictions = np.where(p_high > 0.5, y_high, y_low)
        
        # Clip predictions to valid range
        predictions = np.clip(predictions, self.target_min, self.target_max)
        
        return predictions
    
    def predict_proba(self, X):
        """Return classification probabilities (useful for analysis)."""
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)
        else:
            preds = self.classifier.predict(X)
            return np.column_stack([1 - preds, preds])

    

def get_data(processed_data_path=config.PROCESSED_DATA_PATH,
             version_name=config.VERSION_NAME,
             include_val=True):
    """Load preprocessed train/val/test data."""
    if include_val:
        datasets = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
    else:
        datasets = ['X_train', 'X_test', 'y_train', 'y_test']
    
    try:
        data = [
            pd.read_csv(f'{processed_data_path}{version_name}_{dataset}.csv')
            for dataset in datasets
        ]
        print(f"✓ Successfully loaded data from {processed_data_path}")
        
        if include_val:
            print(f"  - Train: {data[0].shape}")
            print(f"  - Val:   {data[1].shape}")
            print(f"  - Test:  {data[2].shape}")
        else:
            print(f"  - Train: {data[0].shape}")
            print(f"  - Test:  {data[1].shape}")
        
        return data
    except FileNotFoundError as e:
        print(f"✗ Error loading data: {e}")
        if include_val:
            print(f"  Validation set not found. Trying without validation set...")
            return get_data(processed_data_path, version_name, include_val=False)
        else:
            print(f"  Make sure preprocessing has been run for version: {version_name}")
            raise


def get_classifier(classifier_name: str, **hyperparams):
    """Get classifier for the gating function."""
    random_state = hyperparams.get('random_state', 42)
    
    if classifier_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=random_state, max_iter=1000)
    
    elif classifier_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=hyperparams.get('clf_n_estimators', 100),
            learning_rate=hyperparams.get('clf_learning_rate', 0.1),
            max_depth=hyperparams.get('clf_max_depth', 3),
            random_state=random_state,
            n_jobs=-1
        )
    
    elif classifier_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
        
        return LGBMClassifier(
            n_estimators=hyperparams.get('clf_n_estimators', 100),
            learning_rate=hyperparams.get('clf_learning_rate', 0.1),
            max_depth=hyperparams.get('clf_max_depth', 3),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
    
    elif classifier_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=hyperparams.get('clf_n_estimators', 100),
            max_depth=hyperparams.get('clf_max_depth', None),
            random_state=random_state,
            n_jobs=-1
        )
    
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")


def get_regressor(regressor_name: str, **hyperparams):
    """Get regressor for specialized experts."""
    random_state = hyperparams.get('random_state', 42)
    
    if regressor_name == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    elif regressor_name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=hyperparams.get('reg_alpha', 1.0))
    
    elif regressor_name == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=hyperparams.get('reg_alpha', 1.0), max_iter=10000)

    elif regressor_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=hyperparams.get('reg_n_estimators', 200),
            max_depth=hyperparams.get('reg_max_depth', None),
            min_samples_split=hyperparams.get('reg_min_samples_split', 2),
            random_state=random_state,
            n_jobs=-1
        )

    elif regressor_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=hyperparams.get('reg_n_estimators', 300),
            learning_rate=hyperparams.get('reg_learning_rate', 0.05),
            max_depth=hyperparams.get('reg_max_depth', 6),
            random_state=random_state,
            n_jobs=-1
        )
    
    elif regressor_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
        
        return LGBMRegressor(
            n_estimators=hyperparams.get('reg_n_estimators', 300),
            learning_rate=hyperparams.get('reg_learning_rate', 0.05),
            max_depth=hyperparams.get('reg_max_depth', 6),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
    
    elif regressor_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=hyperparams.get('reg_n_estimators', 200),
            learning_rate=hyperparams.get('reg_learning_rate', 0.1),
            max_depth=hyperparams.get('reg_max_depth', 3),
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")


def calculate_threshold(y_train, method='median', percentile=50, fixed_value=90):
    """Calculate threshold for splitting data into low/high bins."""
    if method == 'median':
        threshold = y_train.median()
    elif method == 'mean':
        threshold = y_train.mean()
    elif method == 'percentile':
        threshold = np.percentile(y_train, percentile)
    elif method == 'fixed':
        threshold = fixed_value
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return threshold


def save_schema(X_train, version_name, model_name):
    """Save feature schema for inference alignment."""
    schema_path = Path(config.MODEL_PATH) / f"{version_name}_{model_name}_schema.json"
    Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    
    schema = {
        "columns": list(X_train.columns),
        "n_features": len(X_train.columns),
        "dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()}
    }
    
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"✓ Saved schema: {schema_path} ({schema['n_features']} features)")
    return schema_path


def evaluate_classifier(moe_model, scaler, X, y, threshold, set_name=""):
    """Evaluate the classifier component."""
    y_values = y.values.ravel() if isinstance(y, (pd.DataFrame, pd.Series)) else y.ravel()
    y_binary = (y_values >= threshold).astype(int)
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    y_pred = moe_model.classifier.predict(X_scaled)
    
    accuracy = accuracy_score(y_binary, y_pred)
    
    if hasattr(moe_model.classifier, 'predict_proba'):
        y_proba = moe_model.classifier.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y_binary, y_proba)
    else:
        auc = None
    
    return {
        f'{set_name}_clf_accuracy': accuracy,
        f'{set_name}_clf_auc': auc if auc else 0.0
    }


def evaluate_model(moe_model, scaler, X_train, y_train, X_val, y_val, X_test, y_test, 
                   threshold, cv_folds=5, has_val=True):
    """Evaluate mixture of experts model on all splits."""
    print("\n" + "="*70)
    print("EVALUATING MIXTURE OF EXPERTS MODEL")
    print("="*70)
    
    # Scale all datasets
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val) if has_val else None
    X_test_scaled = scaler.transform(X_test)
    
    # Train predictions
    y_train_pred = moe_model.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
    }
    
    # Add classifier metrics
    metrics.update(evaluate_classifier(moe_model, scaler, X_train, y_train, threshold, 'train'))
    
    predictions = {
        'train': (y_train, y_train_pred),
    }
    
    # Validation predictions
    if has_val:
        y_val_pred = moe_model.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        metrics.update({
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
        })
        metrics.update(evaluate_classifier(moe_model, scaler, X_val, y_val, threshold, 'val'))
        predictions['val'] = (y_val, y_val_pred)
    
    # Test predictions
    y_test_pred = moe_model.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics.update({
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
    })
    metrics.update(evaluate_classifier(moe_model, scaler, X_test, y_test, threshold, 'test'))
    predictions['test'] = (y_test, y_test_pred)
    
    # Print regression results
    if has_val:
        print(f"\n{'Regression Metric':<20} {'Train':<15} {'Validation':<15} {'Test':<15}")
        print("-" * 65)
        print(f"{'RMSE':<20} {train_rmse:<15.4f} {val_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"{'MAE':<20} {train_mae:<15.4f} {val_mae:<15.4f} {test_mae:<15.4f}")
        print(f"{'R²':<20} {train_r2:<15.4f} {val_r2:<15.4f} {test_r2:<15.4f}")
        
        print(f"\n{'Classifier Metric':<20} {'Train':<15} {'Validation':<15} {'Test':<15}")
        print("-" * 65)
        print(f"{'Accuracy':<20} {metrics['train_clf_accuracy']:<15.4f} {metrics['val_clf_accuracy']:<15.4f} {metrics['test_clf_accuracy']:<15.4f}")
        print(f"{'AUC':<20} {metrics['train_clf_auc']:<15.4f} {metrics['val_clf_auc']:<15.4f} {metrics['test_clf_auc']:<15.4f}")
    else:
        print(f"\n{'Regression Metric':<20} {'Train':<15} {'Test':<15}")
        print("-" * 50)
        print(f"{'RMSE':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"{'MAE':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
        print(f"{'R²':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
        
        print(f"\n{'Classifier Metric':<20} {'Train':<15} {'Test':<15}")
        print("-" * 50)
        print(f"{'Accuracy':<20} {metrics['train_clf_accuracy']:<15.4f} {metrics['test_clf_accuracy']:<15.4f}")
        print(f"{'AUC':<20} {metrics['train_clf_auc']:<15.4f} {metrics['test_clf_auc']:<15.4f}")
    
    print("="*70 + "\n")
    
    return metrics, predictions


def save_artifacts(moe_model, scaler, metrics, predictions, version_name, model_name, has_val=True):
    """Save model, predictions, and metrics to disk."""
    # Save model and scaler together
    model_dir = Path(config.MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{version_name}_{model_name}_moe.joblib"
    
    # Save both scaler and model as a dict
    model_package = {
        'scaler': scaler,
        'model': moe_model
    }
    joblib.dump(model_package, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save results
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train predictions
    y_train, y_train_pred = predictions['train']
    train_results = pd.DataFrame({
        'y_true': y_train.values.flatten(),
        'y_pred': y_train_pred.flatten(),
        'residual': y_train.values.flatten() - y_train_pred.flatten(),
        'abs_residual': np.abs(y_train.values.flatten() - y_train_pred.flatten())
    })
    train_path = results_dir / f"{version_name}_{model_name}_moe_train_predictions.csv"
    train_results.to_csv(train_path, index=False)
    print(f"✓ Train predictions saved: {train_path}")
    
    # Save validation predictions
    if has_val and 'val' in predictions:
        y_val, y_val_pred = predictions['val']
        val_results = pd.DataFrame({
            'y_true': y_val.values.flatten(),
            'y_pred': y_val_pred.flatten(),
            'residual': y_val.values.flatten() - y_val_pred.flatten(),
            'abs_residual': np.abs(y_val.values.flatten() - y_val_pred.flatten())
        })
        val_path = results_dir / f"{version_name}_{model_name}_moe_val_predictions.csv"
        val_results.to_csv(val_path, index=False)
        print(f"✓ Validation predictions saved: {val_path}")
    
    # Save test predictions
    y_test, y_test_pred = predictions['test']
    test_results = pd.DataFrame({
        'y_true': y_test.values.flatten(),
        'y_pred': y_test_pred.flatten(),
        'residual': y_test.values.flatten() - y_test_pred.flatten(),
        'abs_residual': np.abs(y_test.values.flatten() - y_test_pred.flatten())
    })
    test_path = results_dir / f"{version_name}_{model_name}_moe_test_predictions.csv"
    test_results.to_csv(test_path, index=False)
    print(f"✓ Test predictions saved: {test_path}")
    
    # Save metrics summary
    metrics_df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': list(metrics.values())
    })
    metrics_path = results_dir / f"{version_name}_{model_name}_moe_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Metrics saved: {metrics_path}")
    
    return model_path


def train(classifier_name='xgboost',
          regressor_name='xgboost',
          version_name=config.VERSION_NAME,
          threshold_method='fixed',
          threshold_value=90,
          threshold_percentile=50,
          use_soft_gate=True,
          target_min=0,
          target_max=5,
          use_wandb=True,
          **hyperparams):
    """
    Train a Mixture of Experts regression model.
    
    Architecture:
    1. Classifier (gating function): Predicts high vs low rating
    2. Regressor Low: Specialized for low ratings
    3. Regressor High: Specialized for high ratings
    4. Gating: Soft (weighted) or hard (argmax) combination
    
    Args:
        classifier_name: Type of classifier for gating
        regressor_name: Type of regressor for experts
        version_name: Data version identifier
        threshold_method: How to split data ('fixed', 'median', 'mean', 'percentile')
        threshold_value: Fixed threshold value if method='fixed'
        threshold_percentile: Percentile if method='percentile'
        use_soft_gate: Use soft gating (weighted avg) vs hard gating (argmax)
        target_min: Minimum valid target value
        target_max: Maximum valid target value
        use_wandb: Whether to log to Weights & Biases
        **hyperparams: Model-specific hyperparameters
    """
    print("\n" + "="*70)
    print("TRAINING: MIXTURE OF EXPERTS MODEL")
    print("="*70)
    print(f"Version: {version_name}")
    print(f"Classifier: {classifier_name}")
    print(f"Regressor: {regressor_name}")
    print(f"Threshold method: {threshold_method}")
    print(f"Gating: {'Soft' if use_soft_gate else 'Hard'}")
    print(f"Target range: [{target_min}, {target_max}]")
    print(f"Hyperparameters: {hyperparams if hyperparams else 'Default'}")
    print("="*70 + "\n")
    
    # Load data
    data = get_data(version_name=version_name, include_val=True)
    
    if len(data) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = data
        has_val = True
        print("✓ Using train/validation/test split")
    else:
        X_train, X_test, y_train, y_test = data
        X_val, y_val = None, None
        has_val = False
        print("⚠️  No validation set found - using train/test split only")
    
    # Calculate threshold
    threshold = calculate_threshold(
        y_train, 
        method=threshold_method, 
        percentile=threshold_percentile,
        fixed_value=threshold_value
    )
    print(f"\n✓ Calculated threshold: {threshold:.2f}")
    
    # Count samples in each bin
    low_mask = y_train < threshold
    n_low = low_mask.sum()
    n_high = (~low_mask).sum()
    
    # Handle both Series and DataFrame
    if isinstance(n_low, pd.Series):
        n_low = n_low.iloc[0]
    if isinstance(n_high, pd.Series):
        n_high = n_high.iloc[0]
    
    print(f"  Low samples: {n_low} ({100*n_low/len(y_train):.1f}%)")
    print(f"  High samples: {n_high} ({100*n_high/len(y_train):.1f}%)")
    
    # Save schema
    save_schema(X_train, version_name, f"{classifier_name}_{regressor_name}")
    
    # Initialize W&B
    run = None
    if use_wandb:
        try:
            wandb.login()
            run = wandb.init(
                entity='asmazurik-company',
                project=f"capstone_moe",
                name=f"{version_name}_{classifier_name}_{regressor_name}",
                config={
                    "architecture": "mixture_of_experts",
                    "classifier": classifier_name,
                    "regressor": regressor_name,
                    "version": version_name,
                    "threshold": threshold,
                    "threshold_method": threshold_method,
                    "use_soft_gate": use_soft_gate,
                    "target_min": target_min,
                    "target_max": target_max,
                    "has_validation_set": has_val,
                    **hyperparams
                },
                tags=[version_name, "mixture_of_experts", classifier_name, regressor_name]
            )
            print("✓ W&B logging enabled")
        except Exception as e:
            print(f"⚠ W&B initialization failed: {e}")
            print("  Continuing without W&B logging...")
            use_wandb = False
    
    # Build mixture of experts model
    print("\n" + "="*70)
    print("BUILDING MIXTURE OF EXPERTS")
    print("="*70)
    
    # Create scaler
    scaler = StandardScaler()
    
    # Fit scaler and transform training data
    print("\nScaling features...")
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create models that will work with scaled data
    classifier = get_classifier(classifier_name, **hyperparams)
    regressor_low = get_regressor(regressor_name, **hyperparams)
    regressor_high = get_regressor(regressor_name, **hyperparams)
    
    moe_model = MixtureOfExpertsRegressor(
        classifier=classifier,
        regressor_low=regressor_low,
        regressor_high=regressor_high,
        threshold=threshold,
        use_soft_gate=use_soft_gate,
        target_min=target_min,
        target_max=target_max
    )
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # Fit the MoE model with scaled data
    moe_model.fit(X_train_scaled, y_train)
    
    print("\n✓ All models trained successfully")
    
    # Evaluate
    metrics, predictions = evaluate_model(
        moe_model, scaler, X_train, y_train, X_val, y_val, X_test, y_test, 
        threshold, has_val=has_val
    )
    
    # Log to W&B
    if use_wandb and run:
        run.log(metrics)
        
        # Log a summary table for easy comparison
        summary_table = wandb.Table(
            columns=["Split", "RMSE", "MAE", "R²", "Classifier Accuracy", "Classifier AUC"],
            data=[
                ["Train", metrics['train_rmse'], metrics['train_mae'], metrics['train_r2'], 
                 metrics['train_clf_accuracy'], metrics['train_clf_auc']],
                ["Validation", metrics.get('val_rmse', 0), metrics.get('val_mae', 0), 
                 metrics.get('val_r2', 0), metrics.get('val_clf_accuracy', 0), 
                 metrics.get('val_clf_auc', 0)] if has_val else None,
                ["Test", metrics['test_rmse'], metrics['test_mae'], metrics['test_r2'], 
                 metrics['test_clf_accuracy'], metrics['test_clf_auc']]
            ]
        )
        run.log({"metrics_summary": summary_table})
        
        # Log key metrics to summary for easy tracking
        run.summary['final_val_rmse'] = metrics.get('val_rmse', metrics['test_rmse'])
        run.summary['final_test_rmse'] = metrics['test_rmse']
        run.summary['final_val_r2'] = metrics.get('val_r2', metrics['test_r2'])
        run.summary['final_test_r2'] = metrics['test_r2']
        
        print("✓ Metrics logged to W&B")
    
    # Save artifacts
    model_path = save_artifacts(
        moe_model, scaler, metrics, predictions, version_name, 
        f"{classifier_name}_{regressor_name}", has_val
    )
    
    # Finish W&B run
    if use_wandb and run:
        run.finish()
        print("✓ W&B run finished")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved to: {model_path}")
    
    if has_val:
        print(f"\n🎯 Validation RMSE: {metrics['val_rmse']:.4f} (use this for model selection)")
        print(f"   Validation Classifier Accuracy: {metrics['val_clf_accuracy']:.4f}")
        print(f"📊 Test RMSE: {metrics['test_rmse']:.4f} (final evaluation)")
        print(f"   Test Classifier Accuracy: {metrics['test_clf_accuracy']:.4f}")
        print(f"📈 Test R²: {metrics['test_r2']:.4f}")
    else:
        print(f"\n📊 Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"   Test Classifier Accuracy: {metrics['test_clf_accuracy']:.4f}")
        print(f"📈 Test R²: {metrics['test_r2']:.4f}")
    
    print("="*70 + "\n")
    
    return moe_model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a Mixture of Experts regression model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture
    parser.add_argument(
        "--classifier", 
        type=str, 
        default="xgboost",
        choices=["logistic", "xgboost", "lightgbm", "random_forest"],
        help="Classifier type for gating function"
    )
    parser.add_argument(
        "--regressor", 
        type=str, 
        default="xgboost",
        choices=["linear", "ridge", "lasso", "random_forest", "xgboost", "lightgbm", "gradient_boosting"],
        help="Regressor type for expert models"
    )
    
    # Basic arguments
    parser.add_argument(
        "--version-name", 
        type=str, 
        default=config.VERSION_NAME,
        help="Data version name"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    # Threshold settings
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="fixed",
        choices=["fixed", "median", "mean", "percentile"],
        help="Method to calculate threshold for splitting data"
    )
    parser.add_argument(
        "--threshold-value",
        type=float,
        default=90,
        help="Fixed threshold value (used if threshold-method=fixed)"
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=50,
        help="Percentile for threshold (used if threshold-method=percentile)"
    )
    
    # Gating and constraints
    parser.add_argument(
        "--hard-gate",
        action="store_true",
        help="Use hard gating (argmax) instead of soft gating (weighted average)"
    )
    parser.add_argument(
        "--target-min",
        type=float,
        default=0,
        help="Minimum valid target value for clipping"
    )
    parser.add_argument(
        "--target-max",
        type=float,
        default=5,
        help="Maximum valid target value for clipping"
    )
    
    # Classifier hyperparameters (prefixed with clf_)
    parser.add_argument("--clf-n-estimators", type=int, help="Classifier: number of estimators")
    parser.add_argument("--clf-learning-rate", type=float, help="Classifier: learning rate")
    parser.add_argument("--clf-max-depth", type=int, help="Classifier: max depth")
    
    # Regressor hyperparameters (prefixed with reg_)
    parser.add_argument("--reg-alpha", type=float, help="Regressor: regularization (ridge/lasso)")
    parser.add_argument("--reg-n-estimators", type=int, help="Regressor: number of estimators")
    parser.add_argument("--reg-learning-rate", type=float, help="Regressor: learning rate")
    parser.add_argument("--reg-max-depth", type=int, help="Regressor: max depth")
    parser.add_argument("--reg-min-samples-split", type=int, help="Regressor: min samples split")
    
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Build hyperparameters dict
    hyperparams = {}
    for key in ['clf_n_estimators', 'clf_learning_rate', 'clf_max_depth',
                'reg_alpha', 'reg_n_estimators', 'reg_learning_rate', 
                'reg_max_depth', 'reg_min_samples_split', 'random_state']:
        value = getattr(args, key.replace('-', '_'), None)
        if value is not None:
            hyperparams[key] = value
    
    # Train model
    train(
        classifier_name=args.classifier,
        regressor_name=args.regressor,
        version_name=args.version_name,
        threshold_method=args.threshold_method,
        threshold_value=args.threshold_value,
        threshold_percentile=args.threshold_percentile,
        use_soft_gate=not args.hard_gate,
        target_min=args.target_min,
        target_max=args.target_max,
        use_wandb=not args.no_wandb,
        **hyperparams
    )