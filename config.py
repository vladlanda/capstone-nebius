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

    #LLM
    NEBIUS_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1"
    CHECKPOINT_DIR ="checkpoints/sentiment_analysis"
    BATCH_SIZE = 30
    CONCURRENT_REQUESTS = 20
    TEXT_COLUMNS =  ['description', 'host_about', 'neighborhood_overview']
