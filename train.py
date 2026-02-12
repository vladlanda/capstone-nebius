import argparse
import pandas as pd
import joblib
from pathlib import Path
from config import config
from training.model import get_pipeline
from training.hyperparameters_tuning import find_best_catboost_params
from training.metrics import compute_metrics, save_metrics_summary
from training.prediction import save_predictions
from training.wandb import init_wandb_run

def get_data(processed_data_path=config.PROCESSED_DATA_PATH,
             version_name=config.VERSION_NAME):

    datasets = ['X_train', 'X_test', 'y_train', 'y_test']

    return [
        pd.read_csv(f'{processed_data_path}{version_name}_{data}.csv')
        for data in datasets
    ]


def save_schema(X_train, version_name, model_name):
    import json

    schema_path = Path(config.MODEL_PATH) / f"{version_name}_{model_name}_schema.json"
    Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    schema = {"columns": list(X_train.columns)}
    schema_path.write_text(json.dumps(schema, indent=2))
    print(f"Saved schema: {schema_path} with {len(schema['columns'])} columns")
    return schema_path


def save_model(model, version_name, model_name):
    model_dir = Path(config.MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{version_name}_{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"\nPipeline saved to: {model_path}")
    return model_path

def train(
    model_name="linear",
    version_name=config.VERSION_NAME,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    epochs=config.EPOCHS,
    use_optuna=False,
):

    X_train, X_test, y_train, y_test = get_data()

    save_schema(X_train, version_name, model_name)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    run = init_wandb_run(model_name, version_name, batch_size, learning_rate, epochs)

    if model_name == 'catboost' and use_optuna:
        print("Using Optuna to find best CatBoost parameters...")
        find_best_catboost_params(get_data, n_trials=10)

    pipeline = get_pipeline(model_name)
    print("Training model...")
    pipeline.fit(X_train,y_train)
    print("Saving model...")
    save_model(pipeline, version_name, model_name)
    print("Done")

    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)
    save_predictions(y_train, y_train_pred, y_test, y_test_pred, version_name, model_name)

    metrics = compute_metrics(y_train, y_train_pred, y_test, y_test_pred)
    run.log(metrics)
    save_metrics_summary(metrics, version_name, model_name)

    run.finish()


def parse_args():
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

    return parser.parse_args()


def main():
    args = parse_args()

    train(
        model_name=args.model,
        version_name=args.version_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        use_optuna=args.optuna
    )



if __name__ == '__main__':
    main()
