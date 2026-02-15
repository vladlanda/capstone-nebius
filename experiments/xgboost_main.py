import os
import random
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
from sklearn.neighbors import KernelDensity
import sys

# Set non-interactive backend for Matplotlib to avoid Thread/GUI errors
import matplotlib
matplotlib.use('Agg')

# Get the absolute path of the parent directory for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Assuming these helpers exist in your environment
try:
    from preprocess import load_raw_data, convert_numeric_columns
except ImportError:
    # Fallback if the local preprocess module isn't accessible
    def load_raw_data(path): return pd.read_csv(path)
    def convert_numeric_columns(df): return df

# ---------------------------------------------------------
# 1. PREPROCESSING & DATA LOADING
# ---------------------------------------------------------

def clean_and_feature_engineer(df):
    """
    Advanced preprocessing for NYC Airbnb data.
    """
    # 1. Target Cleaning
    df = df.dropna(subset=['review_scores_rating']).copy()

    # 2. Temporal Features
    date_cols = ['host_since', 'first_review', 'last_review', 'last_scraped']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['host_tenure_days'] = (df['last_scraped'] - df['host_since']).dt.days
    df['listing_age_days'] = (df['last_scraped'] - df['first_review']).dt.days
    df['days_since_last_review'] = (df['last_scraped'] - df['last_review']).dt.days

    # 3. Numeric Cleaning
    if df['price'].dtype == 'object':
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

    for col in ['host_response_rate', 'host_acceptance_rate']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('%', '').astype(float)

    # 4. Property Ratios
    df['beds_per_bedroom'] = df['beds'] / (df['bedrooms'].replace(0, 1))
    df['accommodates_per_bedroom'] = df['accommodates'] / (df['bedrooms'].replace(0, 1))

    # 5. Amenity Engineering
    premium_amenities = ['dishwasher', 'washer', 'dryer', 'private entrance', 'coffee maker', 'balcony']
    df['amenities'] = df['amenities'].str.lower()
    df['premium_amenity_count'] = 0
    for amenity in premium_amenities:
        df['premium_amenity_count'] += df['amenities'].str.contains(amenity).fillna(False).astype(int)

    df['total_amenity_count'] = df['amenities'].str.count(',').fillna(0) + 1

    # 6. Feature Selection
    cols_to_keep = [
        'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
        'accommodates', 'bedrooms', 'beds', 'price', 'minimum_nights',
        'host_response_rate', 'host_acceptance_rate', 'instant_bookable',
        'host_tenure_days', 'listing_age_days', 'days_since_last_review',
        'beds_per_bedroom', 'accommodates_per_bedroom',
        'premium_amenity_count', 'total_amenity_count'
    ]

    # Add sentiment scores if they exist in the dataframe
    sentiment_cols = [c for c in df.columns if '_llm_score' in c]
    cols_to_keep.extend(sentiment_cols)

    bool_cols = ['host_is_superhost', 'instant_bookable']
    for col in bool_cols:
        df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0).astype(int)

    X = df[cols_to_keep].fillna(0)
    y = df['review_scores_rating']

    return X, y

def prepare_data_for_xgboost(input_csv):
    df = load_raw_data(input_csv)
    df = convert_numeric_columns(df)
    X, y = clean_and_feature_engineer(df)
    # print(df.head())
    X = X[['description_llm_score', 'host_about_llm_score', 'neighborhood_overview_llm_score']]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)

    return X_train, X_test, y_train, y_test

def load_credentials():
    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY not found.")
    wandb.login()
    return {
        "project": os.getenv("WANDB_PROJECT", "xgboost-airbnb-regression"),
        "entity": os.getenv("WANDB_ENTITY", "asmazurik-company")
    }

def load_data(data_dir, val_size=0.2):
    print(f"Loading data...")
    # Update this path to your actual data source
    X_train_full, X_test_orig, y_train_full, y_test_orig = prepare_data_for_xgboost('./data/raw_llm/')

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=42
    )

    return X_train_raw, X_test_orig, X_val_raw, y_train, y_val, y_test_orig

# ---------------------------------------------------------
# 2. SWEEP & TRAINING
# ---------------------------------------------------------

def get_sweep_config():
    """
    Standard regression sweep config.
    """
    return {
        'method': 'bayes',
        'metric': {'name': 'test_rmse', 'goal': 'minimize'},
        'parameters': {
            'feature_fraction': {'distribution': 'uniform', 'min': 0.5, 'max': 1.0},
            'n_estimators': {'values': [500, 1000, 1500]},
            'learning_rate': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},
            'max_depth': {'distribution': 'int_uniform', 'min': 3, 'max': 10},
            'subsample': {'distribution': 'uniform', 'min': 0.6, 'max': 1.0},
            'colsample_bytree': {'distribution': 'uniform', 'min': 0.4, 'max': 0.8},
            'gamma': {'distribution': 'uniform', 'min': 0, 'max': 10},
            'reg_alpha': {'distribution': 'uniform', 'min': 0.1, 'max': 20},
            'reg_lambda': {'distribution': 'uniform', 'min': 1, 'max': 20}
        }
    }

def create_plots(y_true, y_pred, feature_names, booster):
    plots = {}

    # 1. Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_true.values.flatten() - y_pred.flatten()
    sns.scatterplot(x=y_pred.flatten(), y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plots["residuals_plot"] = wandb.Image(plt)
    plt.close()

    # 2. Actual vs Predicted Plot (R2 Visualization)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal')

    # Calculate R2 for the plot title
    r2 = r2_score(y_true, y_pred)

    # Add diagonal reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted (R²: {r2:.4f})')
    plt.legend()
    plots["predicted_vs_actual_plot"] = wandb.Image(plt)
    plt.close()

    return plots

def train():
    global X_train_all, X_test_all, X_val_all, y_train, y_val, y_test_orig

    with wandb.init() as run:
        config = run.config

        # 1. Random Feature Selection
        all_features = list(X_train_all.columns)
        num_to_select = max(1, int(len(all_features) * config.feature_fraction))
        rng = random.Random(run.id)
        selected_features = rng.sample(all_features, num_to_select)

        run.config.update({"selected_features": selected_features})

        # 2. Run Name
        run.name = f"reg_d{config.max_depth}_lr{config.learning_rate:.3f}_f{config.feature_fraction:.2f}"

        X_train_sub = X_train_all[selected_features]
        X_val_sub = X_val_all[selected_features]
        X_test_sub = X_test_all[selected_features]

        dtrain = xgb.DMatrix(X_train_sub, label=y_train)
        dval = xgb.DMatrix(X_val_sub, label=y_val)
        dtest = xgb.DMatrix(X_test_sub)

        params = {
            "objective": "reg:squarederror", # Back to standard regression
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

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=config.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            callbacks=[WandbCallback(log_model=True)],
            verbose_eval=False,
        )

        y_test_pred = booster.predict(dtest)

        rmse = np.sqrt(mean_squared_error(y_test_orig, y_test_pred))
        mae = mean_absolute_error(y_test_orig, y_test_pred)
        r2 = r2_score(y_test_orig, y_test_pred)

        run.log({
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "best_iteration": booster.best_iteration
        })

        plots = create_plots(y_test_orig, y_test_pred, selected_features, booster)
        run.log(plots)

if __name__ == "__main__":
    settings = load_credentials()
    sweep_config = get_sweep_config()

    X_train_all, X_test_all, X_val_all, y_train, y_val, y_test_orig = load_data('./data/raw_llm/')

    sweep_id = wandb.sweep(sweep_config, project=settings["project"], entity=settings["entity"])
    wandb.agent(sweep_id, function=train, count=40)
