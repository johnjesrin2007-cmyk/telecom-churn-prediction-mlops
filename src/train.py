import os
import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_model(pipeline, X, y):
    
   
    mlflow.set_experiment("telecom-churn-prediction-mlops")

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=kfold, scoring="accuracy"
    )

    print("CV Scores:", cv_scores)
    print("Average CV Accuracy:", np.mean(cv_scores))

    
    param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [0.1, 1, 10]
}

    random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,             
    scoring="accuracy",    
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


   
    with mlflow.start_run():

       
        random_search.fit(X_train, y_train)

        
        best_model = random_search.best_estimator_

        y_pred = best_model.predict(X_test)

       
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

      
        mlflow.log_param("model", " XGBClassifier")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_params(random_search.best_params_)
        mlflow.log_param("num_rows", X.shape[0])
        mlflow.log_param("num_features", X.shape[1])

       
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("CV_Accuracy_mean", np.mean(cv_scores))

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