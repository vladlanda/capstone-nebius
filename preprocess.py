import argparse
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from config import config


def preprocess(raw_data_path=config.RAW_DATA_PATH, 
               drop_duplicate_rows=True,
               handle_column_types=True,
               handle_missing_values=True,
               handle_outliers=True,
               version_name=config.VERSION_NAME,
               split_ratio=config.TEST_SIZE,
               seed=config.RANDOM_SEED,
               processed_data_path=config.PROCESSED_DATA_PATH):
    # IMPORT RAW DATA
    try:
        la = pd.read_csv(f'{raw_data_path}airbnb_la_raw.csv')
        ny = pd.read_csv(f'{raw_data_path}airbnb_ny_raw.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find input files in {raw_data_path}")
        raise

    # CONCAT AND ADD THE CITY COLUMN
    la['city'] = "Los Angeles"
    ny['city'] = "New York"
    airbnb = pd.concat([la, ny], axis=0).reset_index(drop=True).reset_index(names='id')
    print(f"Initial shape: {airbnb.shape}")

    # DROP DUPLICATE ROWS
    if drop_duplicate_rows:
        airbnb = airbnb.drop_duplicates(subset=[c for c in airbnb.columns if c != 'id'], keep='first')
        
        print(f"After dropping duplicates: {airbnb.shape}")

    # # HANDLE MISSING VALUES
    if handle_missing_values:
        # HANDLE MISSING VALUES FOR HOST FEATURES WITH RANDOM SAMPLING IMPUTATION
        mask = (airbnb['host_response_rate'].isna())
        airbnb.loc[mask, 'host_response_rate'] = np.random.choice(
            airbnb.loc[~mask, 'host_response_rate'],
            size=mask.sum(),
            replace=True
        )

        mask = (airbnb['host_response_time'].isna())
        airbnb.loc[mask, 'host_response_time'] = np.random.choice(
            airbnb.loc[~mask, 'host_response_time'],
            size=mask.sum(),
            replace=True
        )

        mask = (airbnb['host_acceptance_rate'].isna())
        airbnb.loc[mask, 'host_acceptance_rate'] = np.random.choice(
            airbnb.loc[~mask, 'host_acceptance_rate'],
            size=mask.sum(),
            replace=True
        )

        # DROP MISSING VALUES OF TARGET COLUMN
        airbnb = airbnb.dropna(subset=['first_review', 'last_review','review_scores_rating'], how='any')

        # EXTRACT MISSING VALUES IN 'bathrooms' FROM 'bathrooms_text'
        bathrooms_extracted = (airbnb["bathrooms_text"]
            .str.extract(r"(\d+\.?\d*)")
            .astype(float)[0])
        airbnb["bathrooms"] = airbnb["bathrooms"].fillna(bathrooms_extracted)
        airbnb = airbnb.drop('bathrooms_text', axis=1)
        
        print(f"After handling missing values: {airbnb.shape}")

    # # HANDLE COLUMN DATA TYPES
    if handle_column_types:
        # DATE COLUMNS
        date_cols = ['last_scraped', 'host_since', 'first_review', 'last_review']
        for col in date_cols:
            airbnb[col] = pd.to_datetime(airbnb[col])
        
        # Extract numeric features from dates
        reference_date = pd.to_datetime('2024-01-01')
        
        for col in date_cols:
            # Days since reference date
            airbnb[f'{col}_days_since'] = (airbnb[col] - reference_date).dt.days
            # Drop original date column
            airbnb = airbnb.drop(col, axis=1)

        # BOOLEAN COLUMNS
        bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'instant_bookable']
        for col in bool_cols:
            airbnb[col] = airbnb[col].map({'t':True , 'f': False}).astype(float)

        # ORDINAL COLUMNS
        ordinal_cols = ['host_response_time']
        response_time_order = {
            'within an hour': 0,
            'within a few hours': 1,
            'within a day': 2,
            'a few days or more': 3
        }
        airbnb['host_response_time'] = airbnb['host_response_time'].map(response_time_order).astype(float)      

        #NUMERIC COLUMNS
        numeric_cols = [
            'host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'host_total_listings_count',
            'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'minimum_nights',
            'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
            'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'estimated_occupancy_l365d',
            'review_scores_rating'
        ]
        for col in numeric_cols:
            if col in ['host_response_rate', 'host_acceptance_rate']:
                # Remove %, $ and commas, then convert to float
                airbnb[col] = (
                    airbnb[col]
                    .astype(str)
                    .str.replace('[\$,%,]', '', regex=True)
                    .replace('', np.nan)
                    .astype(float) / 100
                )

            elif col in ['price']:
                # Remove %, $ and commas, then convert to float
                airbnb[col] = (
                    airbnb[col]
                    .astype(str)
                    .str.replace('[\$,%,]', '', regex=True)
                    .replace('', np.nan)
                    .astype(float)
                )
            else:
                airbnb[col] = airbnb[col].astype(float)

        # LIST COLUMNS
        # Parse the list columns first
        list_cols = ['host_verifications', 'amenities']

        for col in list_cols:
            def parse_list(x):
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

            airbnb[col] = airbnb[col].apply(parse_list)

        # Host verifications is fine - only 3 categories
        mlb = MultiLabelBinarizer(sparse_output=True)
        sparse_matrix = mlb.fit_transform(airbnb['host_verifications'])
        dummies = pd.DataFrame.sparse.from_spmatrix(
            sparse_matrix,
            columns=[f"host_verifications_{c}" for c in mlb.classes_],
            index=airbnb.index
        )
        airbnb = airbnb.drop(columns=['host_verifications'])
        airbnb = pd.concat([airbnb, dummies], axis=1)

        # For amenities, keep only the TOP 50 most common ones
        from collections import Counter

        all_amenities = [item for sublist in airbnb['amenities'] for item in sublist]
        amenity_counts = Counter(all_amenities)
        top_50_amenities = set([amenity for amenity, count in amenity_counts.most_common(50)])

        # Filter to only keep top 50
        airbnb['amenities_filtered'] = airbnb['amenities'].apply(
            lambda x: [item for item in x if item in top_50_amenities]
        )

        mlb = MultiLabelBinarizer(sparse_output=True)
        sparse_matrix = mlb.fit_transform(airbnb['amenities_filtered'])
        dummies = pd.DataFrame.sparse.from_spmatrix(
            sparse_matrix,
            columns=[f"amenities_{c}" for c in mlb.classes_],
            index=airbnb.index
        )

        airbnb = airbnb.drop(columns=['amenities', 'amenities_filtered'])
        airbnb = pd.concat([airbnb, dummies], axis=1)

        # CATEGORICAL COLUMNS
        # 1. Property type - group rare categories into "Other"
        property_counts = airbnb['property_type'].value_counts()
        rare_properties = property_counts[property_counts < 1000].index
        airbnb['property_type'] = airbnb['property_type'].replace(rare_properties, 'Other')

        # 2. One-hot encode all three columns
        categorical_cols = ['property_type', 'room_type', 'city']
        airbnb = pd.get_dummies(airbnb, columns=categorical_cols, drop_first=True, sparse=False, dtype=int)


    # # OUTLIERS
    if handle_outliers:
        nights_cols = ['minimum_nights','maximum_nights','minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm']
        for col in nights_cols:
            airbnb.loc[airbnb[col]>365,col] = np.nan

    # # Keep only  numeric fields
    airbnb = airbnb.select_dtypes(include=[np.number])

    # # Fill NaN
    airbnb = airbnb.fillna(0)

    # # SPLIT
    X = airbnb.drop(['review_scores_rating', 'id'], axis=1)
    y = airbnb['review_scores_rating']

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

    # Also save full processed data
    full_filepath = f'{processed_data_path}{version_name}_processed.csv'
    airbnb.to_csv(full_filepath, index=False)
    print(f"Saved full processed data: {airbnb.shape} to {full_filepath}")

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
                handle_outliers=args.handle_outliers, 
                version_name=args.version_name,
                split_ratio=args.split_ratio,
                seed=args.seed,
                processed_data_path=args.processed_data_path
                )