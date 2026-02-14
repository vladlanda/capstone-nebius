import argparse
import pandas as pd
import numpy as np
import ast
import re
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from config import config
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import logging
import sys
from glob import glob

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import asyncio
import shutil
import random

logging.basicConfig(level=logging.INFO)

def load_raw_data(raw_data_path):
    """Load and concatenate Airbnb data from multiple CSV files."""
    csv_files = glob(os.path.join(raw_data_path, '*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_path}")

    dfs = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Add city column based on filename
            filename = os.path.basename(csv_file).lower()

            if 'la' in filename:
                df['city'] = 'Los Angeles'
            elif 'ny' in filename:
                df['city'] = 'New York'
            else:
                # Default city name from filename
                df['city'] = os.path.splitext(filename)[0]

            dfs.append(df)
            logging.info(f"Loaded {len(df)} rows from {os.path.basename(csv_file)} (city: {df['city'].iloc[0]})")
        except Exception as e:
            logging.error(f"Failed to load {csv_file}: {e}")
            continue

    if not dfs:
        raise ValueError("No data was successfully loaded from CSV files")

    # Concatenate all dataframes
    if len(dfs) > 1:
        airbnb = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        airbnb = dfs[0].copy()

    # Reset index and create id column AFTER all operations
    airbnb = airbnb.reset_index(drop=True)
    airbnb.insert(0, 'id', airbnb.index)

    logging.info(f"Total shape after loading: {airbnb.shape}")
    logging.info(f"Cities in dataset: {airbnb['city'].value_counts().to_dict()}")

    return airbnb

def remove_duplicates(df):
    """Remove duplicate rows from dataframe."""
    initial_shape = df.shape
    # Create a copy of the id column before dropping duplicates
    df = df.drop_duplicates(subset=[c for c in df.columns if c not in ['id']], keep='first')
    # Reassign sequential IDs after dropping duplicates
    df = df.reset_index(drop=True)
    df['id'] = df.index

    logging.info(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows: {initial_shape} → {df.shape}")
    return df


def impute_missing_values(df, seed=None, fitted_params=None):
    """Handle missing values through imputation and dropping.

    Args:
        df: Input dataframe
        seed: Random seed
        fitted_params: If provided, use these sampling distributions instead of fitting

    Returns:
        df: Transformed dataframe
        params: Dictionary of fitted parameters (distributions for each column)
    """
    logging.info(f"Missing values before imputation:\n{df.isna().sum()[df.isna().sum() > 0]}")

    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    params = fitted_params or {}

    # Random sampling imputation for host features
    for col in ['host_response_rate', 'host_response_time', 'host_acceptance_rate']:
        if col not in df.columns:
            continue
        mask = df[col].isna()
        if mask.sum() > 0:
            if fitted_params is None:
                # Fit: store the distribution
                non_missing = df.loc[~mask, col].dropna()
                if len(non_missing) > 0:
                    params[col] = non_missing.values.copy()
                    df.loc[mask, col] = np.random.choice(
                        params[col],
                        size=mask.sum(),
                        replace=True
                    )
                    logging.info(f"Fit and imputed {mask.sum()} missing values in '{col}'")
            else:
                # Transform: use fitted distribution
                if col in fitted_params and len(fitted_params[col]) > 0:
                    df.loc[mask, col] = np.random.choice(
                        fitted_params[col],
                        size=mask.sum(),
                        replace=True
                    )
                    logging.info(f"Imputed {mask.sum()} missing values in '{col}' using fitted params")

    # Drop rows with missing critical date columns
    critical_cols = ['first_review', 'last_review', 'review_scores_rating']
    initial_len = len(df)
    df = df.dropna(subset=critical_cols, how='any')
    logging.info(f"Dropped {initial_len - len(df)} rows with missing date columns")

    # Extract bathrooms from bathrooms_text
    if 'bathrooms_text' in df.columns:
        bathrooms_extracted = (df["bathrooms_text"]
            .str.extract(r"(\d+\.?\d*)")
            .astype(float)[0])
        df.loc[:, "bathrooms"] = df.loc[:, "bathrooms"].fillna(bathrooms_extracted)
        df = df.drop('bathrooms_text', axis=1)
        logging.info("Extracted bathroom counts from bathrooms_text")

    logging.info(f"After handling missing values: {df.shape}")
    return df, params

def impute_price_knn(df, fitted_params=None):
    """
    Impute missing price values using KNN based on location and property features.
    Uses: latitude, longitude, bedrooms, bathrooms, accommodates, beds, property_type

    Args:
        df: Input dataframe
        fitted_params: If provided, contains fitted KNNImputer from training

    Returns:
        df: Transformed dataframe
        params: Dictionary containing fitted KNNImputer
    """
    from sklearn.impute import KNNImputer

    logging.info("Starting KNN imputation for price...")

    # Check if price has missing values
    missing_count = df['price'].isna().sum()
    if missing_count == 0:
        logging.info("No missing prices found, skipping KNN imputation")
        return df, fitted_params or {}

    logging.info(f"Missing prices: {missing_count} ({df['price'].isna().mean()*100:.2f}%)")

    # Define features for KNN imputation
    knn_features = ['latitude', 'longitude', 'bedrooms', 'bathrooms', 'accommodates', 'beds']

    # Check if property_type columns exist (created after encode_categorical_columns)
    property_type_cols = [col for col in df.columns if col.startswith('property_type_')]
    if property_type_cols:
        knn_features.extend(property_type_cols)
        logging.info(f"Found {len(property_type_cols)} property_type columns for KNN")
    else:
        logging.warning("property_type columns not found - using basic features only")

    # Add price to features list
    all_features = knn_features + ['price']

    # Check which features exist in dataframe
    available_features = [f for f in all_features if f in df.columns]
    missing_features = [f for f in all_features if f not in df.columns]

    if missing_features:
        logging.warning(f"Missing features for KNN: {missing_features}")

    if len(available_features) < 3:
        logging.error(f"Not enough features for KNN imputation. Available: {available_features}")
        logging.info("Falling back to median imputation")
        median_price = df['price'].median()
        df.loc[df['price'].isna(), 'price'] = median_price
        return df, fitted_params or {}

    # Create subset for imputation
    df_knn = df[available_features].copy()

    # Convert to numeric (handle any string values)
    for col in df_knn.columns:
        df_knn[col] = pd.to_numeric(df_knn[col], errors='coerce')

    # Check if we have enough non-missing data
    rows_with_price = df_knn['price'].notna().sum()
    if rows_with_price < 10:
        logging.error(f"Not enough non-missing prices for KNN ({rows_with_price} rows). Using median instead.")
        median_price = df['price'].median()
        df.loc[df['price'].isna(), 'price'] = median_price
        return df, fitted_params or {}

    try:
        if fitted_params is None:
            # Fit mode: create and fit imputer on this data
            n_neighbors = min(5, rows_with_price - 1)
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            df_imputed = pd.DataFrame(
                imputer.fit_transform(df_knn),
                columns=df_knn.columns,
                index=df_knn.index
            )
            params = {'knn_imputer': imputer, 'knn_features': available_features}
            logging.info(f"✓ Fitted KNN imputer using {len(available_features)-1} features")
        else:
            # Transform mode: use fitted imputer
            imputer = fitted_params.get('knn_imputer')
            fitted_features = fitted_params.get('knn_features', available_features)

            # Ensure same features are used
            if set(fitted_features) != set(available_features):
                # Align to fitted features
                df_knn = df_knn.reindex(columns=fitted_features, fill_value=0)

            df_imputed = pd.DataFrame(
                imputer.transform(df_knn),
                columns=df_knn.columns,
                index=df_knn.index
            )
            params = fitted_params
            logging.info(f"✓ Applied fitted KNN imputer")

        # Update only the price column in original dataframe
        df['price'] = df_imputed['price']
        logging.info(f"  Features used: {[f for f in available_features if f != 'price']}")

    except Exception as e:
        logging.error(f"KNN imputation failed: {e}")
        logging.info("Falling back to median imputation")
        median_price = df['price'].median()
        df.loc[df['price'].isna(), 'price'] = median_price
        params = fitted_params or {}

    return df, params

def convert_date_columns(df):
    """Convert date columns to days since reference date."""
    date_cols = ['last_scraped', 'host_since', 'first_review', 'last_review']
    reference_date = pd.to_datetime('2026-01-01')

    for col in date_cols:
        if col not in df.columns:
            logging.warning(f"Date column '{col}' not found in dataframe")
            continue
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_days_since'] = (df[col] - reference_date).dt.days
        df = df.drop(col, axis=1)
        logging.info(f"Converted '{col}' to days since reference date")

    return df


def convert_boolean_columns(df):
    """Convert boolean columns from 't'/'f' to numeric."""
    bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'instant_bookable']
    for col in bool_cols:
        if col not in df.columns:
            logging.warning(f"Boolean column '{col}' not found in dataframe")
            continue
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
    if 'host_response_time' in df.columns:
        df['host_response_time'] = df['host_response_time'].map(response_time_order).astype(float)
    return df


def convert_numeric_columns(df):
    """Convert numeric columns to proper numeric types."""
    numeric_cols = [
        'host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'host_total_listings_count',
        'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'minimum_nights',
        'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
        'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'estimated_occupancy_l365d',
    ]

    for col in numeric_cols:
        if col not in df.columns:
            continue

        if col in ['host_response_rate', 'host_acceptance_rate']:
            # Handle percentage values - check if already numeric
            if df[col].dtype in ['float64', 'int64']:
                # Already numeric, check if needs division by 100
                if df[col].max() > 1.0:
                    df[col] = df[col] / 100
            else:
                # String format: remove %, $ and commas, then convert
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace('[\$,%,]', '', regex=True)
                    .replace('', np.nan)
                    .astype(float) / 100
                )
        elif col in ['price']:
            # Handle price - check if already numeric
            if df[col].dtype in ['float64', 'int64']:
                # Already numeric, no conversion needed
                pass
            else:
                # String format: remove $, commas
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace('[\$,%,]', '', regex=True)
                    .replace('', np.nan)
                    .astype(float)
                )
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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

def encode_list_columns(df, fitted_params=None):
    """Encode list columns (host_verifications and amenities) using MultiLabelBinarizer.

    Args:
        df: Input dataframe
        fitted_params: If provided, contains fitted MultiLabelBinarizers

    Returns:
        df: Transformed dataframe
        params: Dictionary of fitted MultiLabelBinarizers
    """
    list_cols = ['host_verifications', 'amenities']

    # Parse list columns
    for col in list_cols:
        if col not in df.columns:
            logging.warning(f"List column '{col}' not found in dataframe")
            continue
        df[col] = df[col].apply(parse_list)

    params = fitted_params or {}

    # Encode host_verifications
    if 'host_verifications' in df.columns:
        if fitted_params is None:
            # Fit mode
            mlb = MultiLabelBinarizer(sparse_output=False)
            encoded = mlb.fit_transform(df['host_verifications'])
            params['host_verifications_mlb'] = mlb
            logging.info(f"Fitted host_verifications encoder with {len(mlb.classes_)} classes")
        else:
            # Transform mode
            mlb = fitted_params['host_verifications_mlb']
            encoded = mlb.transform(df['host_verifications'])
            logging.info(f"Applied fitted host_verifications encoder")

        dummies = pd.DataFrame(
            encoded,
            columns=[f"host_verifications_{c}" for c in mlb.classes_],
            index=df.index,
            dtype='int8'
        )
        df = df.drop(columns=['host_verifications'])
        df = pd.concat([df, dummies], axis=1)

    # Encode amenities (keep only top 50)
    if 'amenities' in df.columns:
        if fitted_params is None:
            # Fit mode: determine top 50 amenities
            all_amenities = [item for sublist in df['amenities'] for item in sublist]
            amenity_counts = Counter(all_amenities)
            top_50_amenities = set([amenity for amenity, count in amenity_counts.most_common(50)])
            params['top_50_amenities'] = top_50_amenities
            logging.info(f"Identified top 50 amenities from training data")
        else:
            # Transform mode: use fitted top 50
            top_50_amenities = fitted_params['top_50_amenities']
            logging.info(f"Using fitted top 50 amenities")

        df['amenities_filtered'] = df['amenities'].apply(
            lambda x: [item for item in x if item in top_50_amenities]
        )

        if fitted_params is None:
            # Fit mode
            mlb = MultiLabelBinarizer(sparse_output=False)
            encoded = mlb.fit_transform(df['amenities_filtered'])
            params['amenities_mlb'] = mlb
            logging.info(f"Fitted amenities encoder with {len(mlb.classes_)} classes")
        else:
            # Transform mode
            mlb = fitted_params['amenities_mlb']
            encoded = mlb.transform(df['amenities_filtered'])
            logging.info(f"Applied fitted amenities encoder")

        dummies = pd.DataFrame(
            encoded,
            columns=[f"amenities_{c}" for c in mlb.classes_],
            index=df.index,
            dtype='int8'
        )

        df = df.drop(columns=['amenities', 'amenities_filtered'])
        df = pd.concat([df, dummies], axis=1)

    return df, params

def neighborhood_extraction(df, fitted_params=None):
    """
    Extract neighborhood names from text columns using frequency-based approach.
    """

    logging.info("\n=== Frequency-Based Neighborhood Extraction ===")

    # Define columns to search in
    text_columns = ['neighborhood_overview', 'description', 'name']
    available_columns = [col for col in text_columns if col in df.columns]

    if not available_columns:
        logging.warning("None of the text columns found: neighborhood_overview, description, name")
        df['extracted_neighborhood'] = np.nan
        return df, fitted_params or {}

    logging.info(f"Using columns: {available_columns}")

    params = fitted_params or {}

    if fitted_params is None:
        # FIT MODE: Extract neighborhoods from this dataset
        logging.info("FIT MODE: Extracting capitalized phrases from all texts...")

        all_phrases = []
        for text in df['neighborhood_overview'].dropna():
            # Match capitalized phrases (up to 2 words)
            matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\b', str(text))
            all_phrases.extend(matches)

        # Count frequencies
        logging.info("Counting phrase frequencies...")
        phrase_counts = Counter(all_phrases)

        # Filter out common non-neighborhood words
        stop_phrases = {
            'The', 'This', 'Our', 'Welcome', 'You', 'Your', 'We', 'Great', 'Beautiful',
            'Amazing', 'Perfect', 'Best', 'Enjoy', 'Come', 'Located', 'Close', 'Near',
            'Walking', 'Minutes', 'Easy', 'Access', 'Public', 'Transportation',
            'Restaurants', 'Shops', 'Bars', 'Coffee', 'Grocery', 'Subway', 'Metro',
            'Bus', 'Train', 'Airport', 'Beach', 'Park', 'Street', 'Avenue', 'Boulevard',
            'Room', 'Bedroom', 'Bathroom', 'Kitchen', 'Living', 'Apartment', 'House',
            'Home', 'Space', 'Place', 'Guest', 'Private', 'Shared', 'Cozy', 'Modern',
            'Los Angeles', 'There', 'New York', 'Central Park', 'City', 'Universal Studios',
            'Walk', 'Just', 'With', 'Times Square', 'Please', 'One', 'Very', 'Quiet',
            'Side', 'Spacious', 'Relax', 'Experience', 'Museum', 'Bed', 'For', 'Art',
            'Center','Starbucks', 'Pier', 'Whole Foods', 'Angeles', 'Building', 'Downtown',
            'Approx', 'Prospect Park', 'Fame', 'Grand Central', 'Empire State', 'Disneyland',
            'And', 'Market', 'Also', 'Garden',
        }

        # Get top neighborhoods (appearing at least 10 times)
        top_neighborhoods = {
            phrase: count for phrase, count in phrase_counts.items()
            if count >= 100 and phrase not in stop_phrases
            and len(phrase) > 2 and not phrase.isdigit()
        }

        logging.info(f"Top 20 neighborhoods by frequency:")
        for phrase, count in list(Counter(top_neighborhoods).most_common(20)):
            logging.info(f"  {phrase}: {count}")

        # Get only top 20 for matching
        top_20_neighborhoods = dict(Counter(top_neighborhoods).most_common(20))
        neighborhoods_lower = {phrase.lower(): phrase for phrase in top_20_neighborhoods.keys()}

        # Store fitted parameters
        params['neighborhoods_lower'] = neighborhoods_lower
        params['top_20_neighborhoods'] = top_20_neighborhoods
        params['stop_phrases'] = stop_phrases

        logging.info(f"✓ Fitted neighborhood extractor with {len(neighborhoods_lower)} neighborhoods")

    else:
        # TRANSFORM MODE: Use fitted neighborhoods
        neighborhoods_lower = fitted_params['neighborhoods_lower']
        top_20_neighborhoods = fitted_params['top_20_neighborhoods']
        logging.info(f"TRANSFORM MODE: Using {len(neighborhoods_lower)} fitted neighborhoods")

    # Match neighborhoods back to listings
    logging.info(f"Matching neighborhoods to listings...")

    def find_neighborhood(row):
        """Search for neighborhood across multiple columns."""
        # Combine all available text columns for this row
        combined_text = ""
        for col in available_columns:
            if col in row.index and pd.notna(row[col]):
                combined_text += " " + str(row[col])

        if not combined_text.strip():
            return np.nan

        text_lower = combined_text.lower()
        matches = []

        for neighborhood_lower, neighborhood_original in neighborhoods_lower.items():
            if neighborhood_lower in text_lower:
                position = text_lower.find(neighborhood_lower)
                # Bonus score if found in name or neighborhood_overview (more reliable)
                position_bonus = 1.5 if 'neighborhood_overview' in combined_text[:position+len(neighborhood_lower)] else 1.0
                score = top_20_neighborhoods[neighborhood_original] * (1000 / (position + 1)) * position_bonus
                matches.append((neighborhood_original, score))

        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0][0]

        return np.nan

    df['extracted_neighborhood'] = df.apply(find_neighborhood, axis=1)

    # Report results
    neighborhood_counts = df['extracted_neighborhood'].value_counts()
    logging.info(f"\nNeighborhood Extraction Results:")
    logging.info(f"  Total unique neighborhoods: {len(neighborhood_counts)}")
    logging.info(f"  Successfully matched: {df['extracted_neighborhood'].notna().sum()} ({df['extracted_neighborhood'].notna().mean()*100:.1f}%)")
    logging.info(f"  Not matched (NaN): {df['extracted_neighborhood'].isna().sum()} ({df['extracted_neighborhood'].isna().mean()*100:.1f}%)")

    if len(neighborhood_counts) > 0:
        logging.info(f"\nTop 10 neighborhoods in this dataset:")
        for neighborhood, count in neighborhood_counts.head(10).items():
            logging.info(f"  {neighborhood}: {count}")

    return df, params

def encode_categorical_columns(df, fitted_params=None, rare_threshold_pct=0.01):
    """
    One-hot encode categorical columns including extracted_neighborhood.

    Args:
        df: Input dataframe
        fitted_params: If provided, contains rare_properties/neighborhoods sets from training
        rare_threshold_pct: Percentage threshold for grouping rare categories (default: 1%)

    Returns:
        df: Transformed dataframe
        params: Dictionary containing rare_properties/neighborhoods sets and column order
    """
    params = fitted_params or {}

    # Handle property_type
    if 'property_type' in df.columns:
        if fitted_params is None:
            # Fit mode: determine rare properties from THIS dataset
            property_counts = df['property_type'].value_counts()
            threshold = int(len(df) * rare_threshold_pct)
            rare_properties = property_counts[property_counts < threshold].index.tolist()
            params['rare_properties'] = rare_properties
            params['property_threshold'] = threshold

            if len(rare_properties) > 0:
                df['property_type'] = df['property_type'].replace(rare_properties, 'Other')
                logging.info(f"Fit: Grouped {len(rare_properties)} rare property types into 'Other' (threshold: {threshold} samples)")
        else:
            # Transform mode: use fitted rare properties
            rare_properties = fitted_params['rare_properties']
            if len(rare_properties) > 0:
                df['property_type'] = df['property_type'].replace(rare_properties, 'Other')
                logging.info(f"Transform: Grouped {len(rare_properties)} rare property types into 'Other'")

    # Handle extracted_neighborhood
    if 'extracted_neighborhood' in df.columns:
        if fitted_params is None:
            # Fit mode: determine rare neighborhoods from THIS dataset
            neighborhood_counts = df['extracted_neighborhood'].value_counts()
            threshold = int(len(df) * rare_threshold_pct)
            rare_neighborhoods = neighborhood_counts[neighborhood_counts < threshold].index.tolist()
            params['rare_neighborhoods'] = rare_neighborhoods
            params['neighborhood_threshold'] = threshold

            if len(rare_neighborhoods) > 0:
                df['extracted_neighborhood'] = df['extracted_neighborhood'].replace(rare_neighborhoods, 'Other')
                logging.info(f"Fit: Grouped {len(rare_neighborhoods)} rare neighborhoods into 'Other' (threshold: {threshold} samples)")
        else:
            # Transform mode: use fitted rare neighborhoods
            rare_neighborhoods = fitted_params.get('rare_neighborhoods', [])
            if len(rare_neighborhoods) > 0:
                df['extracted_neighborhood'] = df['extracted_neighborhood'].replace(rare_neighborhoods, 'Other')
                logging.info(f"Transform: Grouped {len(rare_neighborhoods)} rare neighborhoods into 'Other'")

    # One-hot encode all categorical columns
    categorical_cols = ['property_type', 'room_type', 'extracted_neighborhood']
    existing_cols = [col for col in categorical_cols if col in df.columns]

    if existing_cols:
        logging.info(f"One-hot encoding categorical columns: {existing_cols}")

        if fitted_params is None:
            # FIT MODE: Create dummies and save the column names
            df = pd.get_dummies(df, columns=existing_cols, drop_first=True, dtype='int8')

            # Save the dummy column names for each categorical variable
            dummy_columns = {}
            for col in existing_cols:
                dummy_cols = [c for c in df.columns if c.startswith(f'{col}_')]
                dummy_columns[col] = dummy_cols

            params['dummy_columns'] = dummy_columns
            logging.info(f"Fit: Created dummy columns: {dummy_columns}")

        else:
            # TRANSFORM MODE: Create dummies and align with fitted columns
            df = pd.get_dummies(df, columns=existing_cols, drop_first=True, dtype='int8')

            # Get the expected columns from fit
            expected_dummy_columns = fitted_params.get('dummy_columns', {})

            # For each categorical variable, ensure columns match
            for col, expected_cols in expected_dummy_columns.items():
                current_cols = [c for c in df.columns if c.startswith(f'{col}_')]

                # Add missing columns (fill with 0)
                for exp_col in expected_cols:
                    if exp_col not in df.columns:
                        df[exp_col] = 0
                        logging.info(f"Transform: Added missing column '{exp_col}' (all zeros)")

                # Remove extra columns (categories not seen in training)
                for curr_col in current_cols:
                    if curr_col not in expected_cols:
                        df = df.drop(columns=[curr_col])
                        logging.info(f"Transform: Dropped unexpected column '{curr_col}'")

            # Ensure column order matches
            all_expected_cols = [col for cols in expected_dummy_columns.values() for col in cols]
            non_dummy_cols = [c for c in df.columns if not any(c.startswith(f'{cat}_') for cat in existing_cols)]
            df = df[non_dummy_cols + all_expected_cols]

            logging.info(f"Transform: Aligned columns with fitted data")

    return df, params

def handle_outliers(df):
    """Handle outliers in night-related columns."""
    nights_cols = [
        'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
        'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm'
    ]

    outliers_replaced = 0
    for col in nights_cols:
        if col not in df.columns:
            continue
        mask = df[col] > 365
        outliers_replaced += mask.sum()
        df.loc[mask, col] = np.nan

    logging.info(f"Replaced {outliers_replaced} outlier values (>365 nights) with NaN")
    return df

def engineer_features(df):
    """Create engineered features: price_per_guest and is_professional_host"""
    if 'price' in df.columns and 'accommodates' in df.columns:
        df['price_per_guest'] = (df['price'] / df['accommodates'].replace(0, 1)).astype(float)
        logging.info("Created feature: price_per_guest")

    if 'host_listings_count' in df.columns:
        df['is_professional_host'] = (df['host_listings_count'] > 4).astype(int)
        logging.info("Created feature: is_professional_host")

    return df

def prepare_final_dataset(df, keep_city=False):
    """Keep only numeric fields and impute missing values with column means."""
    # Preserve city column and id before selecting numeric types
    city_col = None
    id_col = None

    if keep_city and 'city' in df.columns:
        city_col = df['city'].copy()

    if 'id' in df.columns:
        id_col = df['id'].copy()

    all_columns = set(df.columns)
    df = df.select_dtypes(include=[np.number])
    only_numeric_columns = set(df.columns)
    dropped_cols = all_columns - only_numeric_columns

    if dropped_cols:
        logging.info(f"Dropped non-numeric columns: {sorted(dropped_cols)}")

    na_rate = df.isna().mean()
    if na_rate[na_rate > 0].any():
        logging.info(f"Columns with missing values:\n{na_rate[na_rate > 0]}")

    logging.info(f"shape before:\n{df.shape}")

    # Impute missing values with column means (no row dropping)
    nan_columns = [col for col in df.columns if df[col].isna().any()]
    if nan_columns:
        logging.info(f"Imputing NaNs in columns: {nan_columns}")
        for col in nan_columns:
            mean_value = df[col].mean(skipna=True)
            if np.isnan(mean_value):
                mean_value = 0.0
                logging.warning(f"Column '{col}' has all NaNs; filling with 0.0")
            df[col] = df[col].fillna(mean_value)

    logging.info(f"shape after mean imputation:\n{df.shape}")

    # Restore preserved columns
    if id_col is not None:
        # Remove id if it exists, then insert at position 0
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        df.insert(0, 'id', id_col)

    if city_col is not None:
        df['city'] = city_col

    return df

def split_data_by_city(df, args):
    """
    Split raw data by city BEFORE preprocessing.
    Train on one city, val/test on another city (50/50 split).
    """
    city_col = 'city'

    if city_col not in df.columns:
        raise ValueError("'city' column not found in dataframe")

    # Get unique cities
    cities = df[city_col].unique()
    logging.info(f"Found cities: {list(cities)}")

    # Determine train and test cities
    if args.train_city:
        train_city = args.train_city
        if train_city not in cities:
            raise ValueError(f"Train city '{train_city}' not found in dataset. Available cities: {list(cities)}")

        test_cities = [c for c in cities if c != train_city]
        if not test_cities:
            raise ValueError(f"Train city '{train_city}' is the only city available")
        test_city = test_cities[0]
    else:
        # Default: first city for training, second for testing
        if len(cities) < 2:
            raise ValueError(f"Need at least 2 cities for city-based split, found {len(cities)}")
        train_city = cities[0]
        test_city = cities[1]

    logging.info(f"Using CITY-BASED split:")
    logging.info(f"  → Train on: {train_city}")
    logging.info(f"  → Val/Test on: {test_city}")

    # Split by city
    train_df = df[df[city_col] == train_city].copy().reset_index(drop=True)
    test_val_df = df[df[city_col] == test_city].copy().reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(f"No data found for training city: {train_city}")
    if len(test_val_df) == 0:
        raise ValueError(f"No data found for test city: {test_city}")

    # Split test city data into validation and test (50/50)
    val_df, test_df = train_test_split(
        test_val_df,
        test_size=0.5,
        random_state=args.seed
    )

    logging.info(f"  → Train samples: {len(train_df)}")
    logging.info(f"  → Val samples: {len(val_df)}")
    logging.info(f"  → Test samples: {len(test_df)}")

    return train_df, val_df, test_df

def split_data_random(df, args):
    """
    Random split of RAW data into train/val/test.
    Returns raw (unprocessed) splits.
    """
    logging.info(f"Using RANDOM split:")
    logging.info(f"  → Train: {(1-args.test_ratio-args.val_ratio)*100:.1f}%")
    logging.info(f"  → Val: {args.val_ratio*100:.1f}%")
    logging.info(f"  → Test: {args.test_ratio*100:.1f}%")

    # Validate ratios
    if args.test_ratio + args.val_ratio >= 1.0:
        raise ValueError(f"test_ratio ({args.test_ratio}) + val_ratio ({args.val_ratio}) must be < 1.0")

    # First split: Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df,
        test_size=args.test_ratio,
        random_state=args.seed
    )

    # Second split: Train vs Val
    val_ratio_adjusted = args.val_ratio / (1 - args.test_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=args.seed
    )

    logging.info(f"  → Train samples: {len(train_df)}")
    logging.info(f"  → Val samples: {len(val_df)}")
    logging.info(f"  → Test samples: {len(test_df)}")

    return train_df, val_df, test_df

async def llm_sentiment_analysis_v2(df, processed_data_path):
    """Asynchronous LLM sentiment analysis with checkpointing."""

    # Try to import from config.py, fallback to environment if not available
    try:
        import config
        CONFIG_MODEL = getattr(config, 'NEBIUS_MODEL', "meta-llama/Meta-Llama-3.1-8B-Instruct")
        CONFIG_BASE_URL = getattr(config, 'NEBIUS_BASE_URL', "https://api.studio.nebius.ai/v1")
        CONFIG_CHECKPOINT_DIR = getattr(config, 'CHECKPOINT_DIR', "checkpoints/sentiment_checkpoints")
        CONFIG_BATCH_SIZE = getattr(config, 'BATCH_SIZE', 15)
        CONFIG_CONCURRENT_REQUESTS = getattr(config, 'CONCURRENT_REQUESTS', 20)
        CONFIG_TEXT_COLUMNS = getattr(config, 'TEXT_COLUMNS', ['description', 'host_about', 'neighborhood_overview'])
    except ImportError:
        CONFIG_MODEL = os.getenv("NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        CONFIG_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
        CONFIG_CHECKPOINT_DIR = ".sentiment_checkpoints"
        CONFIG_BATCH_SIZE = 15
        CONFIG_CONCURRENT_REQUESTS = 20
        CONFIG_TEXT_COLUMNS = ['description']

    async def async_llm_analysis(
        input_df,
        text_columns=None,
        batch_size=None,
        concurrent_requests=None,
        checkpoint_dir=None,
        force_restart=False
    ):
        """
        Analyzes sentiment/quality for multiple text columns using an LLM.
        Saves results as {column}_llm_score.
        """
        load_dotenv()

        # Resolve parameters from config or defaults
        text_columns = text_columns or CONFIG_TEXT_COLUMNS
        batch_size = batch_size or CONFIG_BATCH_SIZE
        concurrent_requests = concurrent_requests or CONFIG_CONCURRENT_REQUESTS
        checkpoint_dir = checkpoint_dir or CONFIG_CHECKPOINT_DIR
        api_key = os.getenv("NEBIUS_API_KEY", "")

        if not api_key:
            raise ValueError("NEBIUS_API_KEY not found in environment variables.")

        client = AsyncOpenAI(api_key=api_key, base_url=CONFIG_BASE_URL)

        # Setup Checkpoint Root
        if force_restart and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

        os.makedirs(checkpoint_dir, exist_ok=True)

        final_results_df = pd.DataFrame(index=input_df.index)

        for col in text_columns:
            if col not in input_df.columns:
                print(f"Skipping {col}: column not found in DataFrame.")
                continue

            print(f"\n--- Processing column: {col} ---")
            col_checkpoint_dir = os.path.join(checkpoint_dir, col)
            os.makedirs(col_checkpoint_dir, exist_ok=True)

            # Identify Processed IDs for this specific column
            processed_data = []
            files = [f for f in os.listdir(col_checkpoint_dir) if f.endswith('.json')]
            for file in files:
                try:
                    with open(os.path.join(col_checkpoint_dir, file), 'r') as f:
                        processed_data.extend(json.load(f))
                except:
                    continue

            processed_ids = {item['id'] for item in processed_data}
            print(f"Resuming '{col}': {len(processed_ids)} rows already processed.")

            # Filter Remaining Work - work on a COPY to avoid modifying original
            work_df = input_df[~input_df.index.isin(processed_ids)].copy()
            work_df[col] = work_df[col].fillna("").str.slice(0, 500)

            if not work_df.empty:
                # Define Internal Async Tasks
                semaphore = asyncio.Semaphore(concurrent_requests)
                system_prompt = (
                    f"You are a hospitality analyst. Analyze the '{col}' of Airbnb listings. "
                    "Return ONLY a JSON object: {'results': [{'id': int, 'score': float}]}. "
                    "Score is -1.0 (unprofessional/sparse) to 1.0 (highly professional/detailed)."
                )

                async def process_batch(batch_df, b_idx, folder):
                    payload = batch_df[[col]].reset_index().rename(columns={'index': 'id'}).to_dict(orient='records')
                    path = os.path.join(folder, f"batch_{b_idx}.json")

                    for attempt in range(5):
                        try:
                            response = await client.chat.completions.create(
                                model=CONFIG_MODEL,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"Analyze: {json.dumps(payload)}"}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.0
                            )
                            results = json.loads(response.choices[0].message.content).get('results', [])
                            with open(path, 'w') as f:
                                json.dump(results, f)
                            return results
                        except Exception as e:
                            logging.warning(f"Attempt {attempt + 1} failed for batch {b_idx}: {e}")
                            await asyncio.sleep((2 ** attempt) + random.random())

                    # All retries failed
                    err_res = [{"id": item['id'], "score": np.nan} for item in payload]
                    with open(path, 'w') as f:
                        json.dump(err_res, f)
                    return err_res

                # Execute Pipeline for this column
                batches = [work_df[i:i + batch_size] for i in range(0, len(work_df), batch_size)]
                start_ts = int(pd.Timestamp.now().timestamp())

                async def sem_task(b, i, f):
                    async with semaphore:
                        return await process_batch(b, i, f)

                tasks = [sem_task(b, start_ts + i, col_checkpoint_dir) for i, b in enumerate(batches)]
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"LLM {col}"):
                    await task

            # Column Consolidation
            all_col_data = []
            for file in os.listdir(col_checkpoint_dir):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(col_checkpoint_dir, file), 'r') as f:
                            all_col_data.extend(json.load(f))
                    except Exception as e:
                        logging.warning(f"Failed to load checkpoint file {file}: {e}")

            if all_col_data:
                col_results = pd.DataFrame(all_col_data).drop_duplicates(subset='id').set_index('id')
                final_results_df[f"{col}_llm_score"] = col_results['score'].reindex(input_df.index)
            else:
                final_results_df[f"{col}_llm_score"] = np.nan

        return final_results_df

    # Run analysis
    scores_df = await async_llm_analysis(input_df=df)
    df = df.join(scores_df)

    # Save results
    full_filepath = os.path.join(processed_data_path, 'airbnb_with_sentiment_analysis.csv')
    os.makedirs(processed_data_path, exist_ok=True)

    try:
        df.to_csv(full_filepath, index=False)
        logging.info(f"Saved sentiment analysis results to {full_filepath}")
    except Exception as e:
        logging.error(f"Failed to save sentiment analysis results: {e}")
        raise

def save_data(df, version_name, name, processed_data_path):
    """Save dataframe to CSV with error handling."""
    filepath = os.path.join(processed_data_path, f'{version_name}_{name}.csv')
    os.makedirs(processed_data_path, exist_ok=True)

    try:
        df.to_csv(filepath, index=False)
        logging.info(f"Saved {name}: {df.shape} to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save {name}: {e}")
        raise

def preprocess_v2_wrapper(args):
    """
    Main preprocessing wrapper that handles both random and city-based splits.
    """
    # Load raw data
    df = load_raw_data(args.raw_data_path)

    # Drop rows with missing target FIRST (critical for all workflows)
    initial_len = len(df)
    df = df.dropna(subset=['review_scores_rating'])
    logging.info(f"Dropped {initial_len - len(df)} rows with missing target variable")

    if len(df) == 0:
        raise ValueError("No data remaining after dropping missing targets")

    logging.info(f"Starting preprocessing with strategy: {args.split_strategy}")

    # Branch based on split strategy
    if args.split_strategy == 'city':
        # CITY SPLIT: Split BEFORE preprocessing, then preprocess with fitted transformers
        train_df_raw, val_df_raw, test_df_raw = split_data_by_city(df, args)

        # Preprocess train set and FIT transformers (including neighborhood extraction)
        logging.info("Preprocessing train set (fitting transformers)...")
        train_df, fitted_params = preprocess_v2(train_df_raw, args, keep_city=False, fitted_params=None)

        # Preprocess val/test sets using FITTED transformers from train
        logging.info("Preprocessing validation set (applying fitted transformers)...")
        val_df, _ = preprocess_v2(val_df_raw, args, keep_city=False, fitted_params=fitted_params)

        logging.info("Preprocessing test set (applying fitted transformers)...")
        test_df, _ = preprocess_v2(test_df_raw, args, keep_city=False, fitted_params=fitted_params)

    else:  # random split
        # RANDOM SPLIT: Split FIRST, then preprocess with fitted transformers
        logging.info("Splitting data randomly into train/val/test...")
        train_df_raw, val_df_raw, test_df_raw = split_data_random(df, args)

        # Preprocess train set and FIT transformers (including neighborhood extraction)
        logging.info("Preprocessing train set (fitting transformers)...")
        train_df, fitted_params = preprocess_v2(train_df_raw, args, keep_city=False, fitted_params=None)

        # Preprocess val/test sets using FITTED transformers from train
        logging.info("Preprocessing validation set (applying fitted transformers)...")
        val_df, _ = preprocess_v2(val_df_raw, args, keep_city=False, fitted_params=fitted_params)

        logging.info("Preprocessing test set (applying fitted transformers)...")
        test_df, _ = preprocess_v2(test_df_raw, args, keep_city=False, fitted_params=fitted_params)

    # Validate target exists in all splits
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if 'review_scores_rating' not in split_df.columns:
            raise ValueError(f"Target 'review_scores_rating' missing in {name} split")
        if split_df['review_scores_rating'].isna().any():
            raise ValueError(f"Target contains NaN values in {name} split")

    # Separate target and id before alignment
    y_train = train_df['review_scores_rating'].copy()
    train_df = train_df.drop(['review_scores_rating'], axis=1, errors='ignore')

    y_val = val_df['review_scores_rating'].copy()
    val_df = val_df.drop(['review_scores_rating'], axis=1, errors='ignore')

    y_test = test_df['review_scores_rating'].copy()
    test_df = test_df.drop(['review_scores_rating'], axis=1, errors='ignore')

    # Check column alignment BEFORE reindexing
    train_cols = set(train_df.columns)
    val_cols = set(val_df.columns)
    test_cols = set(test_df.columns)

    missing_in_val = train_cols - val_cols
    missing_in_test = train_cols - test_cols
    extra_in_val = val_cols - train_cols
    extra_in_test = test_cols - train_cols

    if missing_in_val or extra_in_val:
        logging.warning(f"Val set column mismatch - Missing: {missing_in_val}, Extra: {extra_in_val}")
    if missing_in_test or extra_in_test:
        logging.warning(f"Test set column mismatch - Missing: {missing_in_test}, Extra: {extra_in_test}")

    # Align val and test to train features
    val_df = val_df.reindex(columns=train_df.columns, fill_value=0)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    # Prepare final X matrices (remove id if present)
    X_train = train_df.drop(['id'], axis=1, errors='ignore')
    X_val = val_df.drop(['id'], axis=1, errors='ignore')
    X_test = test_df.drop(['id'], axis=1, errors='ignore')

    # Final validation checks
    assert len(X_train) == len(y_train), f"Row mismatch: X_train={len(X_train)}, y_train={len(y_train)}"
    assert len(X_val) == len(y_val), f"Row mismatch: X_val={len(X_val)}, y_val={len(y_val)}"
    assert len(X_test) == len(y_test), f"Row mismatch: X_test={len(X_test)}, y_test={len(y_test)}"
    assert list(X_train.columns) == list(X_val.columns) == list(X_test.columns), "Column mismatch after alignment"

    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Train set: {len(X_train):,} samples × {len(X_train.columns):,} features")
    print(f"✓ Validation set: {len(X_val):,} samples × {len(X_val.columns):,} features")
    print(f"✓ Test set: {len(X_test):,} samples × {len(X_test.columns):,} features")
    print(f"✓ Split strategy: {args.split_strategy}")
    print(f"✓ Target variable: review_scores_rating")
    print("="*60 + "\n")

    # Save datasets
    logging.info("Saving processed datasets...")

    save_data(X_train, args.version_name, 'X_train', args.processed_data_path)
    save_data(X_val, args.version_name, 'X_val', args.processed_data_path)
    save_data(X_test, args.version_name, 'X_test', args.processed_data_path)
    save_data(y_train, args.version_name, 'y_train', args.processed_data_path)
    save_data(y_val, args.version_name, 'y_val', args.processed_data_path)
    save_data(y_test, args.version_name, 'y_test', args.processed_data_path)

    logging.info(f"✓ All files saved to: {args.processed_data_path}")
    print(f"✓ Files saved with prefix: {args.version_name}_")

def preprocess_v2(df, args, keep_city=False, fitted_params=None):
    """
    Preprocess dataframe with optional city preservation and fitted parameters.

    Args:
        df: Input dataframe
        args: Arguments containing preprocessing flags
        keep_city: If True, preserve city column through preprocessing
        fitted_params: If provided (not None), use these fitted transformers instead of fitting new ones.
                      If None, fit new transformers on this data (training mode).

    Returns:
        df: Preprocessed dataframe
        params: Dictionary of fitted parameters (only populated when fitted_params=None)
    """
    params = fitted_params or {}
    is_training = (fitted_params is None)

    # Remove duplicates (only in training)
    if args.drop_duplicate_rows and is_training:
        df = remove_duplicates(df)

    # Handle missing values (with seed for reproducibility)
    df, params_missing = impute_missing_values(df, seed=args.seed, fitted_params=params.get('missing_values'))
    if is_training:
        params['missing_values'] = params_missing

    # Extract neighborhoods BEFORE column type conversions (needs raw text columns)
    if args.neighborhood_extraction:
        df, params_neighborhood = neighborhood_extraction(df, fitted_params=params.get('neighborhood_extraction'))
        if is_training:
            params['neighborhood_extraction'] = params_neighborhood

    # Handle column types (MUST come after neighborhood extraction, before KNN imputation)
    df = convert_date_columns(df)
    df = convert_boolean_columns(df)
    df = convert_ordinal_columns(df)
    df = convert_numeric_columns(df)

    df, params_list = encode_list_columns(df, fitted_params=params.get('list_encoding'))
    if is_training:
        params['list_encoding'] = params_list

    df, params_cat = encode_categorical_columns(df, fitted_params=params.get('categorical_encoding'))
    if is_training:
        params['categorical_encoding'] = params_cat

    # KNN imputation for price (comes AFTER encoding property_type)
    if args.knn_impute_price:
        df, params_knn = impute_price_knn(df, fitted_params=params.get('knn_imputation'))
        if is_training:
            params['knn_imputation'] = params_knn

    # Add engineered features
    if args.feature_engineering:
        df = engineer_features(df)

    # Handle outliers
    df = handle_outliers(df)

    # Prepare final dataset
    df = prepare_final_dataset(df, keep_city=keep_city)
    df = df.drop(columns=["latitude", "longitude"], errors="ignore")

    return df, params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Airbnb data for ML modeling')

    # Data paths
    parser.add_argument("--raw-data-path", type=str, default=config.RAW_DATA_PATH,
                        help="Path to raw CSV files")
    parser.add_argument("--processed_data_path", type=str, default=config.PROCESSED_DATA_PATH,
                        help="Path to save processed files")
    parser.add_argument("--version-name", type=str, default=config.VERSION_NAME,
                        help="Version name for output files")

    # Preprocessing flags
    parser.add_argument("--drop-duplicate-rows", action='store_true', default=False,
                        help="Remove duplicate rows")
    parser.add_argument("--neighborhood-extraction", action='store_true', default=False,
                        help="Extract neighborhoods from text (integrated into pipeline)")
    parser.add_argument("--knn-impute-price", action='store_true', default=False,
                        help="Use KNN imputation for missing prices")
    parser.add_argument("--feature-engineering", action='store_true', default=False,
                        help="Create engineered features")

    # LLM workflows (separate from main preprocessing)
    parser.add_argument("--llm-sentiment-analysis", action='store_true', default=False,
                        help="Run LLM sentiment analysis (runs separately, then exits)")

    # Split strategy
    parser.add_argument("--split-strategy", type=str, default='random',
                        choices=['random', 'city'],
                        help="Split strategy: 'random' for random split, 'city' for city-based split")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Test set ratio (only for 'random' strategy)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation set ratio (only for 'random' strategy)")
    parser.add_argument("--train-city", type=str, default=None,
                        help="City to use for training (only for 'city' strategy). E.g., 'Los Angeles' or 'New York'")

    # Random seed
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Log arguments
    logging.info("="*60)
    logging.info("PREPROCESSING ARGUMENTS")
    logging.info("="*60)
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    logging.info("="*60)

    # Special case: LLM sentiment analysis runs separately
    if args.llm_sentiment_analysis:
        logging.info("Running LLM sentiment analysis (separate workflow)...")
        df = load_raw_data(args.raw_data_path)
        asyncio.run(llm_sentiment_analysis_v2(df, args.processed_data_path))
        logging.info("LLM sentiment analysis complete!")
        exit(0)

    # Main preprocessing workflow
    preprocess_v2_wrapper(args)

    logging.info("✓ Preprocessing pipeline complete!")


#     python preprocess_new.py `
#   --version-name 'v1_city' `
#   --drop-duplicate-rows `
#   --neighborhood-extraction `
#   --knn-impute-price `
#   --feature-engineering `
#   --split-strategy city `
#   --train-city "Los Angeles" `
#   --seed 42
