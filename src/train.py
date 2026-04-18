import os
import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_model(pipeline, X, y):

    # Reproducibility
    np.random.seed(42)

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("telecom-churn-prediction-mlops")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=kfold, scoring="f1_weighted"
    )

    print("CV Scores:", cv_scores)
    print("Average CV F1:", np.mean(cv_scores))

    # Hyperparameter tuning
    param_dist = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_weighted",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=1
    )

    with mlflow.start_run():

        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_params(random_search.best_params_)

        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("CV_F1_mean", np.mean(cv_scores))

        print("\n------ MODEL METRICS ------")
        print("Accuracy :", acc)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)

        mlflow.sklearn.log_model(best_model, "model")

    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/model.pkl")

    print("\n✅ Best pipeline saved at model/model.pkl")

    return best_model
