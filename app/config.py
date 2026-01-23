class config:
    # Version Control
    VERSION_NAME = 'v1'
    
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
    LEARNING_RATE = 0.001
    EPOCHS = 100
