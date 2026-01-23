import argparse
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

raw_data_path = './data/'

def preprocess(csv_path=raw_data_path, 
               drop_duplicate_rows=True,
               handle_column_types=True,
               handle_missing_values=True,
               handle_outliers=True,
               version_name=None
               ):
    # IMPORT RAW DATA
    try:
        la = pd.read_csv(f'{csv_path}airbnb_la_raw.csv')
        ny = pd.read_csv(f'{csv_path}airbnb_ny_raw.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find input files in {csv_path}")
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

        # BOOLEAN COLUMNS
        bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'instant_bookable']
        for col in bool_cols:
            airbnb[col] = airbnb[col].replace({'t': 1, 'f': 0})

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
        airbnb = pd.get_dummies(airbnb, columns=categorical_cols, drop_first=True, sparse=True, dtype=int)


    # # OUTLIERS
    if handle_outliers:
        nights_cols = ['minimum_nights','maximum_nights','minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm']
        for col in nights_cols:
            airbnb.loc[airbnb[col]>365,col] = np.nan

    airbnb.to_csv(f'{csv_path}airbnb_preprocessed_{version_name}.csv', index=False)
    print(f"Final shape: {airbnb.shape}")
    print(f"Saved to: {csv_path}airbnb_preprocessed_{version_name}.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default=raw_data_path)
    parser.add_argument("--version-name", type=str, default='test')

    args = parser.parse_args()
    preprocess(
        csv_path=args.csv_path,
        version_name=args.version_name
    )