from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def get_model_pipeline(preprocess):
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
    
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])
    
    return pipeline