
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import wandb
from app.utils import set_seed

def train_linear_regression(datapath='data'):

    wandb.login()
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="landavlad",
        # Set the wandb project where this run will be logged.
        project="capstone",
        # Track hyperparameters and run metadata.
        config={
            "model":"linear regression (sklearn)",
        },
    )
    
    X_train = np.load(os.path.join(datapath,'X_train.npy'))
    X_test = np.load(os.path.join(datapath,'X_test.npy'))
    y_train = np.load(os.path.join(datapath,'y_train.npy'))
    y_test = np.load(os.path.join(datapath,'y_test.npy'))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    run.log({"rmse":rmse,"mea":mae,"r2":r2})
    run.finish()



if __name__ == '__main__':
    set_seed()
    train_linear_regression()
