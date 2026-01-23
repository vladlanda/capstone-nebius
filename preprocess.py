
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os

RANDOM_SEED = 42

def set_seed():
    np.random.seed(RANDOM_SEED)

def preprocess_and_split(filepath: str,data_folder='data',split_ratio=0.3):
    df = pd.read_csv(filepath)

    X = df.select_dtypes(include='number').fillna(0).drop(['review_scores_rating','id'], axis=1)
    y = df['review_scores_rating']

    # 2. DATA PREPARATION
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=RANDOM_SEED
    )

    np_file_template = filepath.replace('.csv','')
    logging.debug(np_file_template)
    
    data_np = [X_train, X_test, y_train, y_test]
    filepaths = ['X_train', 'X_test', 'y_train', 'y_test']
    filepaths = [os.path.join(data_folder,f'{d}.npy') for d in filepaths]
    for data,fn in zip(data_np,filepaths):
        np.save(fn,data)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    set_seed()

    filepath = 'data/airbnb_preprocessed.csv'
    preprocess_and_split(filepath=filepath)
