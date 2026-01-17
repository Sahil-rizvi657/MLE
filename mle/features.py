from pathlib import Path
from pyexpat import features
import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from mle.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

def create_features(df):

    features = {}

    for col in ['withdrawal', 'deposit', 'txn_count']:

        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_max'] = df[col].max()
        features[f'{col}_median'] = df[col].median()
        features[f'{col}_sum'] = df[col].sum()



    return features

    

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):

    csv_files = list(Path(input_path).glob("*.csv"))
    features_list = []

    if not csv_files:
        logger.error(f"No CSV files found in {input_path}")
        return

    logger.info(f"Found {len(csv_files)} customer files")

    for file in tqdm(csv_files, desc="Processing customers"):

        try:
            df = pd.read_csv(file)

            df["date"] = pd.to_datetime(df["date"])
            df["customer_id"] = file.stem

            # COMPUTE FEATURES
            # features["customer_id"] = file.stem

            features = create_features(df)

            cust_num = int(file.stem.split("_")[2])

            features["target"] = 0 if cust_num < 5 else 1
            features["customer_id"] = file.stem
            


            features_list.append(features)

        except Exception as e:
            logger.error(f"Failed processing {file}: {e}")

    if features_list:

        all_features_df = pd.DataFrame(features_list)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_features_df.to_csv(output_path, index=False)

        logger.info(f"Features saved to {output_path}")

    
    # -----------------------------------------


if __name__ == "__main__":
    app()
