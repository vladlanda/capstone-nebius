from config import config
from train import train

# Baseline experiment
baseline = {'MODEL_TYPE': 'LinearRegression', 'SCALER_TYPE': 'StandardScaler', 'VERSION_NAME': 'exp_lr_std'}

# Incremental experiments - changing one parameter at a time
experiments = [
    # Baseline
    baseline,
    
    # Experiment 1: Change scaler only (LinearRegression + MinMaxScaler)
    {'MODEL_TYPE': 'LinearRegression', 'SCALER_TYPE': 'MinMaxScaler', 'VERSION_NAME': 'exp_lr_minmax'},
    
    # Experiment 2: Change scaler only (LinearRegression + RobustScaler)
    {'MODEL_TYPE': 'LinearRegression', 'SCALER_TYPE': 'RobustScaler', 'VERSION_NAME': 'exp_lr_robust'},
    
    # Experiment 3: Change model only (Ridge + StandardScaler)
    {'MODEL_TYPE': 'Ridge', 'SCALER_TYPE': 'StandardScaler', 'VERSION_NAME': 'exp_ridge_std'},
    
    # Experiment 4: Change scaler only (Ridge + MinMaxScaler)
    {'MODEL_TYPE': 'Ridge', 'SCALER_TYPE': 'MinMaxScaler', 'VERSION_NAME': 'exp_ridge_minmax'},
    
    # Experiment 5: Change scaler only (Ridge + RobustScaler)
    {'MODEL_TYPE': 'Ridge', 'SCALER_TYPE': 'RobustScaler', 'VERSION_NAME': 'exp_ridge_robust'},
    
    # Experiment 6: Change model only (Lasso + StandardScaler)
    {'MODEL_TYPE': 'Lasso', 'SCALER_TYPE': 'StandardScaler', 'VERSION_NAME': 'exp_lasso_std'},
    
    # Experiment 7: Change scaler only (Lasso + RobustScaler)
    {'MODEL_TYPE': 'Lasso', 'SCALER_TYPE': 'RobustScaler', 'VERSION_NAME': 'exp_lasso_robust'},
    
    # Experiment 8: Change model only (RandomForest + None scaler)
    {'MODEL_TYPE': 'RandomForest', 'SCALER_TYPE': 'None', 'VERSION_NAME': 'exp_rf_none'},
    
    # Experiment 9: Change model only (GradientBoosting + None scaler)
    {'MODEL_TYPE': 'GradientBoosting', 'SCALER_TYPE': 'None', 'VERSION_NAME': 'exp_gb_none'},
    
    # Experiment 10: Change model only (GradientBoosting + StandardScaler)
    {'MODEL_TYPE': 'GradientBoosting', 'SCALER_TYPE': 'StandardScaler', 'VERSION_NAME': 'exp_gb_std'},
    
    # Experiment 11: Change scaler only (RandomForest + StandardScaler)
    {'MODEL_TYPE': 'RandomForest', 'SCALER_TYPE': 'StandardScaler', 'VERSION_NAME': 'exp_rf_std'},
]

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {exp['VERSION_NAME']}")
    print(f"{'='*60}")
    
    # Temporarily update config
    original_model = config.MODEL_TYPE
    original_scaler = config.SCALER_TYPE
    original_version = config.VERSION_NAME
    
    config.MODEL_TYPE = exp['MODEL_TYPE']
    config.SCALER_TYPE = exp['SCALER_TYPE']
    config.VERSION_NAME = exp['VERSION_NAME']
    
    # Run training
    try:
        train(version_name=exp['VERSION_NAME'])
    except Exception as e:
        print(f"Error in {exp['VERSION_NAME']}: {e}")
    
    # Restore config
    config.MODEL_TYPE = original_model
    config.SCALER_TYPE = original_scaler
    config.VERSION_NAME = original_version

print("\nAll experiments completed!")

