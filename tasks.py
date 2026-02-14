from invoke import task

@task
def preprocess_sentiment_analysis_keep_raw(c,preprocess_args = None):
    if preprocess_args is None: preprocess_args = ''
    cmd = (
        f'python preprocess.py '
        f'--llm-sentiment-analysis '
        f'--processed_data_path "./data/raw_llm/" '
        f'{preprocess_args}'
    )
    c.run(cmd,pty=True)


@task
def xgboost_experiment(c):

    cmd = (
        f'python experiments/xgboost_main.py'
    )
    c.run(cmd,pty=True)


@task
def train_xgboost(c):
    cmd = (
        f'python train.py --model xgboost'
    )
    c.run(cmd,pty=True)


@task
def train_catboost(c):
    cmd = (
        f'python train.py --model catboost'
    )
    c.run(cmd,pty=True)

@task
def train(c):
    cmd = (
        f'python train.py --model catboost'
    )
    c.run(cmd,pty=True)


@task
def train_catboost_optuna(c):
    cmd = (
        f'python train.py --model catboost --optuna'
    )
    c.run(cmd,pty=True)

@task
def preprocess(c):
    cmd = (
        f'python preprocess.py --drop-duplicate-rows --handle-outliers'
    )
    c.run(cmd,pty=True)

@task
def server(c):
    cmd = (
        f'streamlit run server.py -- --model xgboost --handle-outliers'
    )
    c.run(cmd,pty=True)

# invoke preprocess-mixture-of-experts
@task
def preprocess_mixture_of_experts(c):
    cmd = (
        'python preprocess_new.py '
        '--version-name v1_random_top80 '
        '--drop-duplicate-rows '
        '--neighborhood-extraction '
        '--knn-impute-price '
        '--feature-engineering '
        '--handle-outliers '
        '--split-strategy random '
        '--test-ratio 0.15 '
        '--val-ratio 0.15 '
        '--seed 42'
    )
    c.run(cmd,pty=True)



# invoke train-mixture-of-experts
@task
def train_mixture_of_experts(c):
    cmd = (
        'python train_mix_of_experts.py '
        '--threshold-method percentile '
        '--threshold-percentile 75 '
        '--clf-n-estimators 200 '
        '--reg-n-estimators 500 '
        '--reg-learning-rate 0.01 '
        '--version-name v1_random_top80'
    )
    c.run(cmd,pty=True)


# @task
# def run_all(c):
#     cmd = (
#         'invoke preprocess '
#         'invoke train_xgboost '
#         'invoke server'
#     )
#     c.run(cmd,pty=True)


@task
def predict(c):
    cmd = (
        'python predict.py '
        '--input data/raw_city3/airbnb_city3_x.csv '
        '--output /tmp/city3_pred.csv'
    )
    c.run(cmd, pty=True)


@task
def evaluate(c):
    cmd = (
        'python evaluate.py '
        '--pred /tmp/city3_pred.csv '
        '--gt data/raw_city3/airbnb_city3_y.csv '
        '--name city3'
    )
    c.run(cmd, pty=True)


@task
def predict_on_test(c):
    cmd = (
        'python predict.py '
        '--input /Users/almaz/nebius_academy/capstone/capstone-nebius/data/processed/v1_X_test.csv '
        '--output /tmp/test_pred.csv'
    )
    c.run(cmd, pty=True)


@task
def evaluate_on_test(c):
    cmd = (
        'python evaluate.py '
        '--pred /tmp/test_pred.csv '
        '--gt /Users/almaz/nebius_academy/capstone/capstone-nebius/data/processed/v1_y_test.csv '
        '--name test_set'
    )
    c.run(cmd, pty=True)
