import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import wandb
import optuna
from config import config
import joblib

## TODO hyperparameters: batch_size, learning_rate, epochs.
config.BATCH_SIZE
config.LEARNING_RATE
config.EPOCHS
## TODO save model to /models config.MODEL_PATH
## TODO save predictions of train and test to results config.RESULTS_PATH

def get_data(processed_data_path=config.PROCESSED_DATA_PATH,
             version_name=config.VERSION_NAME):

    datasets = ['X_train', 'X_test', 'y_train', 'y_test']

    return [
        pd.read_csv(f'{processed_data_path}{version_name}_{data}.csv')
        for data in datasets
    ]

def _load_catboost_best_params():
    params_path = Path(config.MODEL_PATH) / "catboost_best_params.json"
    if params_path.exists():
        import json
        return json.loads(params_path.read_text())
    return None


def find_best_catboost_params(n_trials=5):
    X_train, X_test, y_train, y_test = get_data()
    base_params = _load_catboost_best_params()
    if base_params is None:
        base_params = {
            "iterations": 800,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "rsm": 0.8,
        }

    def _clamp(val, low, high):
        return max(low, min(high, val))

    def objective(trial):
        from catboost import CatBoostRegressor

        params = {
            "iterations": trial.suggest_int(
                "iterations",
                _clamp(int(base_params["iterations"] * 0.7), 300, 1500),
                _clamp(int(base_params["iterations"] * 1.3), 300, 1500),
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                _clamp(base_params["learning_rate"] * 0.7, 0.01, 0.1),
                _clamp(base_params["learning_rate"] * 1.3, 0.01, 0.1),
            ),
            "depth": trial.suggest_int(
                "depth",
                _clamp(base_params["depth"] - 2, 4, 10),
                _clamp(base_params["depth"] + 2, 4, 10),
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                _clamp(base_params["l2_leaf_reg"] * 0.5, 1.0, 10.0),
                _clamp(base_params["l2_leaf_reg"] * 2.0, 1.0, 10.0),
            ),
            "subsample": trial.suggest_float(
                "subsample",
                _clamp(base_params["subsample"] * 0.8, 0.6, 1.0),
                _clamp(base_params["subsample"] * 1.2, 0.6, 1.0),
            ),
            "rsm": trial.suggest_float(
                "rsm",
                _clamp(base_params["rsm"] * 0.8, 0.5, 1.0),
                _clamp(base_params["rsm"] * 1.2, 0.5, 1.0),
            ),
            "loss_function": "RMSE",
            "verbose": False,
            "random_seed": 42,
        }

        cb = CatBoostRegressor(**params)
        cb.fit(X_train, y_train.values.ravel())

        test_pred = cb.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        return test_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    params_path = Path(config.MODEL_PATH) / "catboost_best_params.json"
    Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    import json
    params_path.write_text(json.dumps(best_params, indent=2))

    return None


def _build_pipeline(model, use_scaler=True):
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def load_linear_model():
    from sklearn.linear_model import LinearRegression
    return _build_pipeline(LinearRegression())


def load_ridge_model():
    from sklearn.linear_model import Ridge
    return _build_pipeline(Ridge(alpha=1.0))


def load_random_forest_model():
    from sklearn.ensemble import RandomForestRegressor
    return _build_pipeline(RandomForestRegressor(n_estimators=200))


def load_xgboost_model():
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
    )
    return _build_pipeline(model)


def load_catboost_model():
    from catboost import CatBoostRegressor
    catboost_params = _load_catboost_best_params()
    if catboost_params is None:
        raise ValueError("catboost_best_params.json not found; run Optuna tuning first")
    return CatBoostRegressor(
        **catboost_params,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )


def get_pipeline(model_name: str):
    """
    Returns a full pipeline based on CLI argument.
    """

    if model_name == "linear":
        return load_linear_model()

    if model_name == "ridge":
        return load_ridge_model()

    if model_name == "random_forest":
        return load_random_forest_model()

    if model_name == "xgboost":
        return load_xgboost_model()

    if model_name == "catboost":
        return load_catboost_model()

    raise ValueError(f"Unknown model: {model_name}")

def train(
    model_name="linear",
    version_name=config.VERSION_NAME,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    epochs=config.EPOCHS,
    use_optuna=False,
):

    X_train, X_test, y_train, y_test = get_data()

    # Save schema for inference alignment
    import json
    schema_path = Path(config.MODEL_PATH) / f"{version_name}_{model_name}_schema.json"
    Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    schema = {"columns": list(X_train.columns)}
    schema_path.write_text(json.dumps(schema, indent=2))
    print(f"Saved schema: {schema_path} with {len(schema['columns'])} columns")

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Training with version: {version_name}")

    # Initialize wandb
    wandb.login()
    run = wandb.init(
        entity='asmazurik-company',
        project=f"capstone_train_{model_name}",
        name=f"{version_name}_{model_name}_regression",
        config={
            "model": f"{model_name} regression (sklearn)",
            "version": version_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
        },
    )


    if model_name == 'catboost' and use_optuna:
        print("Using Optuna to find best CatBoost parameters...")
        find_best_catboost_params(n_trials=10)

    print("Training model...")
    pipeline = get_pipeline(model_name)
    pipeline.fit(X_train,y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)
    model_to_save = pipeline


    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Log metrics
    print(f"\nTrain Metrics - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
    print(f"Test Metrics - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

    run.log({
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    })

    # Save model to /models
    model_dir = Path(config.MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{version_name}_{model_name}.joblib"
    joblib.dump(model_to_save, model_path)
    print(f"\nPipeline saved to: {model_path}")

    # Save predictions to /results
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save train predictions
    train_results = pd.DataFrame({
        'y_true': y_train.values.flatten(),
        'y_pred': y_train_pred.flatten(),
        'residual': y_train.values.flatten() - y_train_pred.flatten()
    })
    train_results_path = results_dir / f"{version_name}_{model_name}_train_predictions.csv"
    train_results.to_csv(train_results_path, index=False)
    print(f"Train predictions saved to: {train_results_path}")

    # Save test predictions
    test_results = pd.DataFrame({
        'y_true': y_test.values.flatten(),
        'y_pred': y_test_pred.flatten(),
        'residual': y_test.values.flatten() - y_test_pred.flatten()
    })
    test_results_path = results_dir / f"{version_name}_{model_name}_test_predictions.csv"
    test_results.to_csv(test_results_path, index=False)
    print(f"Test predictions saved to: {test_results_path}")

    # Save metrics summary
    metrics_summary = pd.DataFrame({
        'metric': ['rmse', 'mae', 'r2'],
        'train': [train_rmse, train_mae, train_r2],
        'test': [test_rmse, test_mae, test_r2]
    })
    metrics_path = results_dir / f"{version_name}_{model_name}_metrics.csv"
    metrics_summary.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    run.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--version-name", type=str, default=config.VERSION_NAME)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)

    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "ridge", "random_forest", "xgboost", "catboost"],
        help="Choose which model to train",
    )
    parser.add_argument(
        "--optuna", action="store_true", help="Enable Optuna tuning for CatBoost"
    )


    args = parser.parse_args()

    train(
        model_name=args.model,
        version_name=args.version_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        use_optuna=args.optuna
    )
