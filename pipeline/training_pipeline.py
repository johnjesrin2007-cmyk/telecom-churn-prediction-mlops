import os
import sys
import mlflow
from prefect import flow


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("telechom-churn-prediction-mlops")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.preprocess import preprocess_data, get_preprocessor
from src.train import train_model
from src.model import get_model_pipeline  


@flow(name="ML Training Pipeline")
def training_pipeline():

    data_path = "data/raw.csv"

    
    X, y = preprocess_data(
        path=data_path,
        training=True,
        target_col="Churn"  
    )

    # -------------------------
    # 🧠 BUILD PREPROCESSOR
    # -------------------------
    preprocess = get_preprocessor(X)

    # -------------------------
    # 🤖 BUILD MODEL PIPELINE
    # -------------------------
    pipeline = get_model_pipeline(preprocess)

    # -------------------------
    # 🎯 TRAIN MODEL
    # -------------------------
    trained_pipeline = train_model(pipeline, X, y)

    return trained_pipeline


# -------------------------
# ▶️ RUN
# -------------------------
if __name__ == "__main__":
    training_pipeline()