import os
import numpy as np
import pandas as pd
import xgboost as xgb
import wandb
from wandb.integration.xgboost import WandbCallback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Set non-interactive backend for Matplotlib to avoid Thread/GUI errors
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------

def load_credentials():
    """
    Loads environment variables and logs into W&B.
    """
    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError(
            "WANDB_API_KEY not found in environment variables. "
            "Please create a .env file with WANDB_API_KEY=your_key_here"
        )
    wandb.login()
    return {
        "project": os.getenv("WANDB_PROJECT", "capstone-xgboost-optimization"),
        "entity": os.getenv("WANDB_ENTITY", "asmazurik-company")
    }

def load_data(data_dir, val_size=0.2):
    """
    Reads CSV files and prepares scaled X_train, X_test, X_val, y_train, y_test, y_val.
    Splits the original training set to create a validation set.
    """
    print(f"Loading data from {data_dir}...")
    
    # Load files based on the directory structure provided
    X_train_full = pd.read_csv(os.path.join(data_dir, "v1_X_train.csv"))
    y_train_full = pd.read_csv(os.path.join(data_dir, "v1_y_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "v1_X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "v1_y_test.csv"))

    # Split training into train and validation BEFORE scaling to prevent leakage
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full, 
        y_train_full, 
        test_size=val_size, 
        random_state=42
    )

    # Initialize and apply StandardScaler
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train_raw)
    
    # Transform validation and test sets using the training fit
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames to keep column names for plotting/XGBoost
    X_train = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val_raw.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Prepare scaler parameters for logging
    scaler_params = {
        "scaler_means": dict(zip(X_train_raw.columns, scaler.mean_)),
        "scaler_scales": dict(zip(X_train_raw.columns, scaler.scale_))
    }

    print(f"Data loaded and scaled successfully:")
    print(f" - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val, scaler_params

def get_sweep_config():
    """
    Returns the hyperparameter sweep configuration.
    """
    return {
        'method': 'bayes',
        'metric': {
          'name': 'val-rmse',
          'goal': 'minimize'   
        },
        'parameters': {
            'n_estimators': { 'values': [100,200,500,700,1000] },
            'learning_rate': { 'distribution': 'uniform', 'min': 0.01, 'max': 0.5 },
            'max_depth': { 'distribution': 'int_uniform', 'min': 10, 'max': 100 },
            'subsample': { 'distribution': 'uniform', 'min': 0.5, 'max': 1.0 },
            'colsample_bytree': { 'distribution': 'uniform', 'min': 0.5, 'max': 1.0 },
            'gamma': { 'distribution': 'uniform', 'min': 0, 'max': 5 },
            'reg_alpha': { 'distribution': 'uniform', 'min': 0, 'max': 10 },
            'reg_lambda': { 'distribution': 'uniform', 'min': 1, 'max': 10 }
        }
    }

def create_plots(y_true, y_pred, feature_names, booster):
    """
    Generates standard regression diagnostic plots for W&B logging.
    """
    plots = {}
    
    # Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_true.values.flatten() - y_pred.flatten()
    sns.scatterplot(x=y_pred.flatten(), y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plots["residuals_plot"] = wandb.Image(plt)
    plt.close()

    # Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min().item(), y_true.max().item()], [y_true.min().item(), y_true.max().item()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plots["prediction_error_plot"] = wandb.Image(plt)
    plt.close()

    # Feature Importance (Weight)
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(booster, max_num_features=15, importance_type='weight')
    plt.title('Feature Importance (Weight)')
    plots["feature_importance_weight"] = wandb.Image(plt)
    plt.close()

    return plots

# ---------------------------------------------------------
# 2. DEFINE TRAINING FUNCTION
# ---------------------------------------------------------

def train():
    """
    Training function using native XGBoost API.
    """
    global X_train, X_test, X_val, y_train, y_test, y_val, scaler_params

    with wandb.init() as run:
        config = run.config
        
        # Log scaler parameters to the run configuration for traceability
        run.config.update(scaler_params)
        
        # Create a custom run name based on hyperparameters
        run_name = "_".join([
            f"n{config.n_estimators}",
            f"lr{config.learning_rate:.3f}",
            f"d{config.max_depth}",
            f"sub{config.subsample:.2f}",
            f"col{config.colsample_bytree:.2f}",
            f"g{config.gamma:.2f}",
            f"a{config.reg_alpha:.2f}",
            f"l{config.reg_lambda:.2f}"
        ])
        run.name = run_name

        # Convert data to DMatrix for native API
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Map hyperparameters to native parameters
        params = {
            "objective": "reg:squarederror",
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "gamma": config.gamma,
            "alpha": config.reg_alpha,
            "lambda": config.reg_lambda,
            "tree_method": "hist",
            "eval_metric": "rmse",
            "random_state": 42
        }

        # Train using native API
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=config.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=20,
            callbacks=[WandbCallback(log_model=True)],
            verbose_eval=False
        )

        # Evaluation on Test Set
        y_test_pred = booster.predict(dtest)
        
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        # Log metrics
        run.log({
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "best_iteration": booster.best_iteration
        })

        # Generate and log diagnostic plots
        diagnostic_plots = create_plots(y_test, y_test_pred, X_train.columns, booster)
        run.log(diagnostic_plots)

# ---------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    # Load settings and credentials
    settings = load_credentials()
    sweep_config = get_sweep_config()
    
    # Resolve directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'processed')
    
    # Load, scale the data, and get scaler parameters
    X_train, X_test, X_val, y_train, y_test, y_val, scaler_params = load_data(data_dir=data_path)
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=settings["project"], 
        entity=settings["entity"]
    )

    # Run the sweep agent
    wandb.agent(sweep_id, function=train, count=20)