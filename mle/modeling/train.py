from pathlib import Path
from xml.parsers.expat import model
import pandas as pd

from loguru import logger
import sklearn
from tqdm import tqdm
import typer
import yaml

from mle.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

app = typer.Typer()

def train_model(X: pd.DataFrame, y: pd.Series, params: dict):
    """
    Train a machine learning model based on specified parameters.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        params: Dictionary of modeling parameters
    
    Returns:
        Trained model
    """

    model_type = params.get("model_type", "random_forest")
    
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=params.get("n_estimators", 100))
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from dvclive import Live

    with Live() as live:

        logger.info(f"Training {model_type} model...")
        model.fit(X_train, y_train)
        logger.info("Model training complete.")

        pred = model.predict(X_test)
        auc = roc_auc_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        live.log_metric("auc", auc)
        live.log_metric("f1", f1)
    
    return model

def save_model(model, model_path: Path):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    import joblib

    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----

    home_dir = PROJ_ROOT
    params_file = home_dir / "params.yaml"
    logger.info(f"Project root: {home_dir}")
    params = yaml.safe_load(open(params_file))["modeling"]

    df = pd.read_csv(features_path)
    logger.info(f"Features loaded from {features_path}")

    X = df.drop(columns=params["columns_to_drop"], axis=1)
    y = df[params["target"]]

    model = train_model(X, y, params)
    save_model(model, model_path)



    # -----------------------------------------


if __name__ == "__main__":
    app()
