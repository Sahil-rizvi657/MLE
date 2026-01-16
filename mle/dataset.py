from pathlib import Path
import zipfile

import pandas as pd
import typer
import openpyxl 
import yaml

from config import INTERIM_DATA_DIR, RAW_DATA_DIR, PROJ_ROOT

app = typer.Typer()


def load_data(input_path):
   
    excel_name = "bank.xlsx"

    if isinstance(input_path, str):
        input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"The specified path does not exist: {input_path}")

    if input_path.suffix == '.zip':
        with zipfile.ZipFile(input_path, 'r') as z:
            with z.open(excel_name) as f:
                df = pd.read_excel(f, engine='openpyxl')
                print(df.head())
                return df
    else:
        df = pd.read_excel(input_path, engine='openpyxl')
        print(df.head())
        return df





@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR/"dataset.zip",
    output_path: Path = INTERIM_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    home_dir = PROJ_ROOT
    print(f"Home directory is set to: {home_dir}")
    params_file = home_dir.as_posix() + "/params.yaml"
    params = yaml.safe_load(open(params_file))["dataset"]
    print(f"Parameters loaded: {params}")

    df = load_data(input_path)
    df.to_csv(output_path, index=False)
    # -----------------------------------------


if __name__ == "__main__":
    app()
