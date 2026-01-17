from pathlib import Path
import zipfile

import pandas as pd
import typer
import openpyxl 
import yaml

from config import INTERIM_DATA_DIR, RAW_DATA_DIR, PROJ_ROOT

app = typer.Typer()

def load_data(input_path):

    df = pd.read_csv(input_path)
    print(f"Data loaded from {input_path} with shape {df.shape}")
    return df
   
    
def rename_columns(df, params):
    
    print(params["rename_columns"])

    df = df.rename(columns=params["rename_columns"])
    return df

def clean_data(df):
    df = df.dropna()
    
    df = df.drop_duplicates()
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df['withdrawal'] = pd.to_numeric(df['withdrawal'], errors='coerce').fillna(0)
    
    df['deposit'] = pd.to_numeric(df['deposit'], errors='coerce').fillna(0)

    df['balance'] = pd.to_numeric(df['balance'], errors='coerce').fillna(0)

    df.reset_index(drop=True, inplace=True)

    return df

def prepare_data(df, Output_path):

    print("Preparing data for each customer...")

    df['date'] = pd.to_datetime(df['date'])

    unique_customers = df['customer_id'].unique()
    print(unique_customers)

    for customer in unique_customers:

        customer_data = df[df['customer_id'] == customer].copy()

        daily = (
            customer_data
            .groupby('date')
            .agg({
                'withdrawal': 'sum',
                'deposit': 'sum',
                'customer_id': 'count'   # number of rows = txn count
            })
            .rename(columns={'customer_id': 'txn_count'})
            .reset_index()
            .sort_values('date')
        )

        daily.to_csv(Output_path/f"customer_{customer}_daily.csv", index=False)



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR/"dataset.csv",
    output_path: Path = INTERIM_DATA_DIR,
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    home_dir = PROJ_ROOT
    print(f"Home directory is set to: {home_dir}")

    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["dataset"]
    print(f"Parameters loaded: {params}")

    df = load_data(input_path)

    df = rename_columns(df, params)
    print("Columns renamed.")

    df = clean_data(df)

    prepare_data(df, output_path)


    # -----------------------------------------


if __name__ == "__main__":
    app()
