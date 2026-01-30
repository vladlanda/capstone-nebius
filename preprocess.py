import argparse
import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from config import config


def load_raw_data(raw_data_path):
    """Load and concatenate LA and NY Airbnb data."""
    try:
        la = pd.read_csv(f'{raw_data_path}airbnb_la_raw.csv')
        ny = pd.read_csv(f'{raw_data_path}airbnb_ny_raw.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find input files in {raw_data_path}")
        raise

    la['city'] = "Los Angeles"
    ny['city'] = "New York"
    airbnb = pd.concat([la, ny], axis=0).reset_index(drop=True).reset_index(names='id')
    print(f"Initial shape: {airbnb.shape}")

    return airbnb


def remove_duplicates(df):
    """Remove duplicate rows from dataframe."""
    df = df.drop_duplicates(subset=[c for c in df.columns if c != 'id'], keep='first')
    print(f"After dropping duplicates: {df.shape}")
    return df


def impute_missing_values(df):
    """Handle missing values through imputation and dropping."""
    # Random sampling imputation for host features
    for col in ['host_response_rate', 'host_response_time', 'host_acceptance_rate']:
        mask = df[col].isna()
        if mask.sum() > 0:
            df.loc[mask, col] = np.random.choice(
                df.loc[~mask, col],
                size=mask.sum(),
                replace=True
            )

    # Drop rows with missing target values
    df = df.dropna(subset=['first_review', 'last_review', 'review_scores_rating'], how='any')

    # Extract bathrooms from bathrooms_text
    bathrooms_extracted = (df["bathrooms_text"]
        .str.extract(r"(\d+\.?\d*)")
        .astype(float)[0])
    df["bathrooms"] = df["bathrooms"].fillna(bathrooms_extracted)
    df = df.drop('bathrooms_text', axis=1)

    print(f"After handling missing values: {df.shape}")
    return df


def convert_date_columns(df):
    """Convert date columns to days since reference date."""
    date_cols = ['last_scraped', 'host_since', 'first_review', 'last_review']
    reference_date = pd.to_datetime('2024-01-01')

    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_days_since'] = (df[col] - reference_date).dt.days
        df = df.drop(col, axis=1)

    return df


def convert_boolean_columns(df):
    """Convert boolean columns from 't'/'f' to numeric."""
    bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'instant_bookable']
    for col in bool_cols:
        df[col] = df[col].map({'t': True, 'f': False}).astype(float)
    return df


def convert_ordinal_columns(df):
    """Convert ordinal columns to numeric."""
    response_time_order = {
        'within an hour': 0,
        'within a few hours': 1,
        'within a day': 2,
        'a few days or more': 3
    }
    df['host_response_time'] = df['host_response_time'].map(response_time_order).astype(float)
    return df


def convert_numeric_columns(df):
    """Convert numeric columns to proper numeric types."""
    numeric_cols = [
        'host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'host_total_listings_count',
        'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'minimum_nights',
        'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
        'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'estimated_occupancy_l365d',
        'review_scores_rating'
    ]

    for col in numeric_cols:
        if col in ['host_response_rate', 'host_acceptance_rate']:
            # Remove %, $ and commas, then convert to float and divide by 100
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('[\$,%,]', '', regex=True)
                .replace('', np.nan)
                .astype(float) / 100
            )
        elif col in ['price']:
            # Remove %, $ and commas, then convert to float
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('[\$,%,]', '', regex=True)
                .replace('', np.nan)
                .astype(float)
            )
        else:
            df[col] = df[col].astype(float)

    return df


def parse_list(x):
    """Parse string representations of lists."""
    if pd.isna(x) or x == '' or x == '[]':
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return []


def encode_list_columns(df):
    """Encode list columns (host_verifications and amenities) using MultiLabelBinarizer."""
    list_cols = ['host_verifications', 'amenities']

    # Parse list columns
    for col in list_cols:
        df[col] = df[col].apply(parse_list)

    # Encode host_verifications
    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_matrix = mlb.fit_transform(df['host_verifications'])
    dummies = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix,
        columns=[f"host_verifications_{c}" for c in mlb.classes_],
        index=df.index
    )
    df = df.drop(columns=['host_verifications'])
    df = pd.concat([df, dummies], axis=1)

    # Encode amenities (keep only top 50)
    all_amenities = [item for sublist in df['amenities'] for item in sublist]
    amenity_counts = Counter(all_amenities)
    top_50_amenities = set([amenity for amenity, count in amenity_counts.most_common(50)])

    df['amenities_filtered'] = df['amenities'].apply(
        lambda x: [item for item in x if item in top_50_amenities]
    )

    mlb = MultiLabelBinarizer(sparse_output=True)
    sparse_matrix = mlb.fit_transform(df['amenities_filtered'])
    dummies = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix,
        columns=[f"amenities_{c}" for c in mlb.classes_],
        index=df.index
    )

    df = df.drop(columns=['amenities', 'amenities_filtered'])
    df = pd.concat([df, dummies], axis=1)

    return df


def encode_categorical_columns(df):
    """One-hot encode categorical columns."""
    # Group rare property types into "Other"
    property_counts = df['property_type'].value_counts()
    rare_properties = property_counts[property_counts < 1000].index
    df['property_type'] = df['property_type'].replace(rare_properties, 'Other')

    # One-hot encode
    categorical_cols = ['property_type', 'room_type', 'city']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, sparse=False, dtype=int)

    return df


def handle_outliers(df):
    """Handle outliers in night-related columns."""
    nights_cols = [
        'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
        'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm'
    ]
    for col in nights_cols:
        df.loc[df[col] > 365, col] = np.nan
    return df


def prepare_final_dataset(df):
    """Keep only numeric fields and fill NaN values."""
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(0)
    return df


def split_and_save_data(df, version_name, split_ratio, seed, processed_data_path):
    """Split data into train/test sets and save to disk."""
    X = df.drop(['review_scores_rating', 'id'], axis=1)
    y = df['review_scores_rating']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=seed
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Save split data
    datasets = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

    for name, data in datasets.items():
        filepath = f'{processed_data_path}{version_name}_{name}.csv'
        data.to_csv(filepath, index=False)
    print(f"Saved split data to {processed_data_path}")

    # Save full processed data
    full_filepath = f'{processed_data_path}{version_name}_processed.csv'
    df.to_csv(full_filepath, index=False)
    print(f"Saved full processed data: {df.shape} to {full_filepath}")


def preprocess(raw_data_path=config.RAW_DATA_PATH,
               drop_duplicate_rows=True,
               handle_column_types=True,
               handle_missing_values=True,
               handle_outliers_flag=True,
               version_name=config.VERSION_NAME,
               split_ratio=config.TEST_SIZE,
               seed=config.RANDOM_SEED,
               processed_data_path=config.PROCESSED_DATA_PATH):
    """Main preprocessing pipeline for Airbnb data."""
    # Load data
    airbnb = load_raw_data(raw_data_path)

    # Remove duplicates
    if drop_duplicate_rows:
        airbnb = remove_duplicates(airbnb)

    # Handle missing values
    if handle_missing_values:
        airbnb = impute_missing_values(airbnb)

    # Handle column types
    if handle_column_types:
        airbnb = convert_date_columns(airbnb)
        airbnb = convert_boolean_columns(airbnb)
        airbnb = convert_ordinal_columns(airbnb)
        airbnb = convert_numeric_columns(airbnb)
        airbnb = encode_list_columns(airbnb)
        airbnb = encode_categorical_columns(airbnb)

    # Handle outliers
    if handle_outliers_flag:
        airbnb = handle_outliers(airbnb)

    # Prepare final dataset
    airbnb = prepare_final_dataset(airbnb)

    # Split and save
    split_and_save_data(airbnb, version_name, split_ratio, seed, processed_data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", type=str, default=config.RAW_DATA_PATH)
    parser.add_argument("--drop-duplicate-rows", type=bool, default=True)
    parser.add_argument("--handle-column-types", type=bool, default=True)
    parser.add_argument("--handle-missing-values", type=bool, default=True)
    parser.add_argument("--handle-outliers", type=bool, default=True)
    parser.add_argument("--version-name", type=str, default=config.VERSION_NAME)
    parser.add_argument("--split-ratio", type=float, default=config.TEST_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--processed_data_path", type=str, default=config.PROCESSED_DATA_PATH)

    args = parser.parse_args()
    preprocess(raw_data_path=args.raw_data_path,
                drop_duplicate_rows=args.drop_duplicate_rows,
                handle_column_types=args.handle_column_types,
                handle_missing_values=args.handle_missing_values,
                handle_outliers_flag=args.handle_outliers,
                version_name=args.version_name,
                split_ratio=args.split_ratio,
                seed=args.seed,
                processed_data_path=args.processed_data_path
                )
