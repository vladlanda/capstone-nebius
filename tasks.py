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
def train_catboost_optuna(c):
    cmd = (
        f'python train.py --model catboost --optuna'
    )
    c.run(cmd,pty=True)

@task
def preprocess(c):
    cmd = (
        f'python preprocess.py --drop-duplicate-rows --handle-column-types --handle-missing-values --handle-outliers'
    )
    c.run(cmd,pty=True)

@task
def server(c):
    cmd = (
        f'streamlit run server.py -- --model xgboost --handle-column-types --handle-missing-values --handle-outliers'
    )
    c.run(cmd,pty=True)

# invoke preprocess-mixture-of-experts
@task
def preprocess_mixture_of_experts(c):
    cmd = (
        'python preprocess_new.py '
        '--version-name v1_random_top80 '
        '--drop-duplicate-rows '
        '--handle-missing-values '
        '--neighborhood-extraction '
        '--handle-column-types '
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
