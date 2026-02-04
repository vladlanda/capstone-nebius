import argparse
import pandas as pd
import numpy as np
import ast
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
# from tqdm import tqdm
from glob import glob

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import asyncio
import shutil
import random

logging.basicConfig(level=logging.INFO)

def load_raw_data(raw_data_path):
    """Load and concatenate LA and NY Airbnb data."""
    # try:
    #     la = pd.read_csv(f'{raw_data_path}airbnb_la_raw.csv')
    #     ny = pd.read_csv(f'{raw_data_path}airbnb_ny_raw.csv')
    # except FileNotFoundError as e:
    #     logging.info(f"Error: Could not find input files in {raw_data_path}")
    #     raise

    csv_files = glob(os.path.join(raw_data_path,'*.csv'))
    dfs = [pd.read_csv(csv) for csv in csv_files]
    # for csv,df in zip(csv_files,dfs):
    #     df['city'] = os.path.basename(csv).split('.')[0]
    
    # la['city'] = "Los Angeles"
    # ny['city'] = "New York"
    # airbnb = pd.concat([la, ny], axis=0).reset_index(drop=True).reset_index(names='id')
    if len(dfs) > 1: 
        airbnb = pd.concat(dfs, axis=0).reset_index(drop=True).reset_index(names='id')
    else:
        airbnb = dfs[0]
    logging.info(f"Initial shape: {airbnb.shape}")

    return airbnb


def remove_duplicates(df):
    """Remove duplicate rows from dataframe."""
    df = df.drop_duplicates(subset=[c for c in df.columns if c != 'id'], keep='first')
    logging.info(f"After dropping duplicates: {df.shape}")
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

    logging.info(f"After handling missing values: {df.shape}")
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
    categorical_cols = ['property_type', 'room_type']
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
        # Save full processed data
    full_filepath = f'{processed_data_path}{version_name}_processed.csv'
    os.makedirs(os.path.dirname(processed_data_path),exist_ok=True)
    df.to_csv(full_filepath, index=False)
    logging.info(f"Saved full processed data: {df.shape} to {full_filepath}")

    if split_ratio <= 0: return

    X = df.drop(['review_scores_rating', 'id'], axis=1)
    # mask = df.columns.str.contains('city')
    # X = df.loc[:, ~mask]
    y = df['review_scores_rating']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=seed
    )

    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

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
    logging.info(f"Saved split data to {processed_data_path}")



def llm_sentiment_analysis(df):

    text_columns = ['description','neighborhood_overview','host_about','amenities']
    load_dotenv()
    # Init client
    client = OpenAI(
        api_key=os.getenv("NEBIUS_API_KEY"),
        base_url="https://api.tokenfactory.nebius.com/v1/",

    )
    
    SYSTEM_PROMPT = """You are an expert analyst of guest perception for rental listings.

                    For EACH listing, estimate how EACH text field would independently influence
                    a typical guest to give a rating between 0 and 5.

                    Rules:
                    - Output scores between 0 and 1
                    - 0.5 = neutral / average
                    - Penalize vague, generic, or missing text
                    - Slightly penalize exaggerated marketing language
                    - Be conservative: most scores should fall between 0.6 and 0.9

                    Return valid JSON only.
                    """
    
    def build_batch_prompt(rows):
        blocks = []

        for i, row in enumerate(rows):
            blocks.append(f"""
                        Listing {i}:
                        Description:
                        {row['description']}

                        Neighborhood Overview:
                        {row['neighborhood_overview']}

                        Host About:
                        {row['host_about']}

                        Amenities:
                        {row['amenities']}
                        """)

        listings_text = "\n".join(blocks)

        return f"""
                Evaluate each listing independently.

                {listings_text}

                Return JSON only as a list, one object per listing, in the SAME order:

                [
                {{
                    "description_llm_score": <float>,
                    "neighborhood_overview_llm_score": <float>,
                    "host_about_llm_score": <float>,
                    "amenities_llm_score": <float>
                }}
                ]
                """

    # lambda function to score a single row
    def score_batch(rows):
        prompt = build_batch_prompt(rows)

        response = client.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content
        return json.loads(content)  # list of dicts
    
    def chunk_df(df, batch_size):
        for start in range(0, len(df), batch_size):
            yield df.iloc[start:start + batch_size]

    BATCH_SIZE = 50
    results = []

    df_tmp = df

    for batch_df in chunk_df(df_tmp, BATCH_SIZE):
        rows = batch_df.to_dict(orient="records")
        batch_scores = score_batch(rows)
        results.extend(batch_scores)

    scores_df = pd.DataFrame(results, index=df_tmp.index)
    df_tmp = pd.concat([df_tmp, scores_df], axis=1)
    

    # df = pd.concat([df, scores_df], axis=1)
    logging.info(df_tmp.head())
    return df_tmp

async def llm_sentiment_analysis_v2(df,processed_data_path):

    # Try to import from config.py, fallback to environment if not available
    try:
        import config
        CONFIG_MODEL = getattr(config, 'NEBIUS_MODEL', "meta-llama/Meta-Llama-3.1-8B-Instruct")
        CONFIG_BASE_URL = getattr(config, 'NEBIUS_BASE_URL', "https://api.studio.nebius.ai/v1")
        CONFIG_CHECKPOINT_DIR = getattr(config, 'CHECKPOINT_DIR', "chekpoints/sentiment_checkpoints")
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
        
        # 1. Setup Checkpoint Root
        if force_restart and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        final_results_df = pd.DataFrame(index=input_df.index)

        for col in text_columns:
            if col not in input_df.columns:
                print(f"Skipping {col}: column not found in DataFrame.")
                continue
                
            print(f"\n--- Processing column: {col} ---")
            col_checkpoint_dir = os.path.join(checkpoint_dir, col)
            if not os.path.exists(col_checkpoint_dir):
                os.makedirs(col_checkpoint_dir)

            # 2. Identify Processed IDs for this specific column
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
            
            # 3. Filter Remaining Work
            work_df = input_df[~input_df.index.isin(processed_ids)].copy()
            work_df[col] = work_df[col].fillna("").str.slice(0, 500)
            
            if not work_df.empty:
                # 4. Define Internal Async Tasks
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
                        except Exception:
                            await asyncio.sleep((2 ** attempt) + random.random())
                    
                    err_res = [{"id": item['id'], "score": np.nan} for item in payload]
                    with open(path, 'w') as f: json.dump(err_res, f)
                    return err_res

                # 5. Execute Pipeline for this column
                batches = [work_df[i:i + batch_size] for i in range(0, len(work_df), batch_size)]
                start_ts = int(pd.Timestamp.now().timestamp())
                
                async def sem_task(b, i, f):
                    async with semaphore: return await process_batch(b, i, f)

                tasks = [sem_task(b, start_ts + i, col_checkpoint_dir) for i, b in enumerate(batches)]
                for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"LLM {col}"):
                    await task

            # 6. Column Consolidation
            all_col_data = []
            for file in os.listdir(col_checkpoint_dir):
                if file.endswith('.json'):
                    with open(os.path.join(col_checkpoint_dir, file), 'r') as f:
                        all_col_data.extend(json.load(f))
            
            if all_col_data:
                col_results = pd.DataFrame(all_col_data).drop_duplicates(subset='id').set_index('id')
                final_results_df[f"{col}_llm_score"] = col_results['score'].reindex(input_df.index)
            else:
                final_results_df[f"{col}_llm_score"] = np.nan

        return final_results_df

        # Parameters like batch_size, checkpoint_dir, etc., are loaded from config.py if not passed here
    # df = df[:100]
    scores_df = await async_llm_analysis(
        input_df=df,
    )
    df = df.join(scores_df)
    full_filepath = f'{processed_data_path}airbnb_with_sentiment_analysis.csv'
    os.makedirs(os.path.dirname(processed_data_path),exist_ok=True)
    df.to_csv(full_filepath, index=False)

def preprocess(raw_data_path=config.RAW_DATA_PATH,
               drop_duplicate_rows=False,
            #    sentiment_analysis = False,
               handle_column_types=False,
               handle_missing_values=False,
               handle_outliers_flag=False,
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

    # if sentiment_analysis:
        # airbnb = llm_sentiment_analysis(airbnb)

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
    # if not keep_raw:
    airbnb = prepare_final_dataset(airbnb)

    # Split and save
    split_and_save_data(airbnb, version_name, split_ratio, seed, processed_data_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-path", type=str, default=config.RAW_DATA_PATH)
    parser.add_argument("--drop-duplicate-rows",    action='store_true', default=False)
    parser.add_argument("--llm-sentiment-analysis", action='store_true', default=False)
    parser.add_argument("--handle-column-types",    action='store_true', default=False)
    parser.add_argument("--handle-missing-values",  action='store_true', default=False)
    parser.add_argument("--handle-outliers",        action='store_true', default=False)
    parser.add_argument("--version-name", type=str, default=config.VERSION_NAME)
    parser.add_argument("--split-ratio", type=float, default=config.TEST_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--processed_data_path", type=str, default=config.PROCESSED_DATA_PATH)

    args = parser.parse_args()
    logging.info(f"Args : {sys.argv}")

    if args.llm_sentiment_analysis:
        df = load_raw_data(args.raw_data_path)
        asyncio.run(llm_sentiment_analysis_v2(df,args.processed_data_path))
        exit(0)

    preprocess(raw_data_path=args.raw_data_path,
                drop_duplicate_rows=args.drop_duplicate_rows,
                # sentiment_analysis = args.llm_sentiment_analysis,
                handle_column_types=args.handle_column_types,
                handle_missing_values=args.handle_missing_values,
                handle_outliers_flag=args.handle_outliers,
                version_name=args.version_name,
                split_ratio=args.split_ratio,
                seed=args.seed,
                processed_data_path=args.processed_data_path,
                )
