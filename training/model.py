from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import config


def load_catboost_best_params():
    params_path = Path(config.MODEL_PATH) / "catboost_best_params.json"
    if params_path.exists():
        import json

        return json.loads(params_path.read_text())
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

    catboost_params = load_catboost_best_params()
    if catboost_params is None:
        raise ValueError("catboost_best_params.json not found; run Optuna tuning first")
    return CatBoostRegressor(
        **catboost_params,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )


def get_pipeline(model_name: str):
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
