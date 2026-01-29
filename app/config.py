class config:
    # Version Control
    VERSION_NAME = 'v2_lr_0.01'
    
    # Seeds
    RANDOM_SEED = 42
    
    # Paths
    RAW_DATA_PATH = './data/raw/'
    PROCESSED_DATA_PATH = './data/processed/'
    MODEL_PATH = './models/'
    RESULTS_PATH = './results/'
    
    # Model Hyperparameters
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 100
    
    # Model and Scaler Selection
    MODEL_TYPE = 'LinearRegression'  # Options: LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting
    SCALER_TYPE = 'StandardScaler'  # Options: StandardScaler, MinMaxScaler, RobustScaler, None