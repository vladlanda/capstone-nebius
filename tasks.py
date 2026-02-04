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