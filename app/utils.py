import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

RANDOM_SEED = 42

def set_seed():
    np.random.seed(RANDOM_SEED)

def get_model(model_type):
    models = {
        'LinearRegression': LinearRegression,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
    }
    return models.get(model_type, LinearRegression)

def get_scaler(scaler_type):
    scalers = {
        'StandardScaler': StandardScaler,
        'MinMaxScaler': MinMaxScaler,
        'RobustScaler': RobustScaler,
        'None': None,
    }
    return scalers.get(scaler_type, StandardScaler)


if __name__ == '__main__':
    pass