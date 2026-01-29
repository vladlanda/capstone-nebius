import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import wandb
from config import config
from utils import get_model, get_scaler

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

def train(version_name=config.VERSION_NAME,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            epochs=config.EPOCHS):

    X_train, X_test, y_train, y_test = get_data()

    print(f"Training with version: {version_name}")
    print(f"Model: {config.MODEL_TYPE}, Scaler: {config.SCALER_TYPE}")

    # Initialize wandb
    wandb.login()
    run = wandb.init(
        project="capstone",
        name=f"{version_name}_{config.MODEL_TYPE.lower()}",
        config={
            "model": config.MODEL_TYPE,
            "scaler": config.SCALER_TYPE,
            "version": version_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
        },
    )

    # Scale features
    scaler_class = get_scaler(config.SCALER_TYPE)
    if scaler_class is not None:
        scaler = scaler_class()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # Train model
    print("Training model...")
    model_class = get_model(config.MODEL_TYPE)
    model = model_class()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

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
    model_path = model_dir / f"{version_name}_{config.MODEL_TYPE.lower()}.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save predictions to /results
    results_dir = Path(config.RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train predictions
    train_results = pd.DataFrame({
        'y_true': y_train.values.flatten(),
        'y_pred': y_train_pred.flatten(),
        'residual': y_train.values.flatten() - y_train_pred.flatten()
    })
    train_results_path = results_dir / f"{version_name}_train_predictions.csv"
    train_results.to_csv(train_results_path, index=False)
    print(f"Train predictions saved to: {train_results_path}")
    
    # Save test predictions
    test_results = pd.DataFrame({
        'y_true': y_test.values.flatten(),
        'y_pred': y_test_pred.flatten(),
        'residual': y_test.values.flatten() - y_test_pred.flatten()
    })
    test_results_path = results_dir / f"{version_name}_test_predictions.csv"
    test_results.to_csv(test_results_path, index=False)
    print(f"Test predictions saved to: {test_results_path}")
    
    # Save metrics summary
    metrics_summary = pd.DataFrame({
        'metric': ['rmse', 'mae', 'r2'],
        'train': [train_rmse, train_mae, train_r2],
        'test': [test_rmse, test_mae, test_r2]
    })
    metrics_path = results_dir / f"{version_name}_metrics.csv"
    metrics_summary.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train linear regression model')
    parser.add_argument("--version-name", type=str, default=config.VERSION_NAME)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    
    args = parser.parse_args()
    
    train(
        version_name=args.version_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )