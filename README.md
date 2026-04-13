# 🚀 Telecom Churn Prediction (MLOps with XGBoost)

An **end-to-end MLOps project** that predicts customer churn in a telecom company using machine learning.
This project demonstrates **production-level ML system design**, including pipelines, experiment tracking, model tuning, cloud deployment, and CI/CD.

---

# 🌍 Live Demo

👉 https://<your-actual-app-name>.onrender.com/docs

> Interactive API documentation (Swagger UI)

---

# 📌 Problem Statement

Customer churn is a critical issue in telecom businesses.
The goal of this project is:

> **Predict whether a customer will leave (churn) or not based on usage and subscription data.**

### 💡 Business Impact:

* Improve customer retention
* Reduce revenue loss
* Enable targeted marketing strategies

---

# 📊 Dataset

* Type: Telecom customer dataset (real-world style)
* Rows: ~2000+
* Features:

  * Demographics (gender, senior citizen)
  * Account info (tenure, contract)
  * Services (internet, streaming, support)
  * Billing (monthly & total charges)

### 🎯 Target:

* `Churn` → Yes / No

---

# 🏗️ Project Architecture

```
telecom-churn-mlops/
│
├── .github/workflows/      → CI/CD pipeline (GitHub Actions)
├── data/                   → Raw dataset (raw.csv)
├── mlruns/                 → MLflow experiment logs
├── model/                  → Saved model artifacts
├── pipeline/               → Training orchestration
├── src/                    → Core ML modules
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
├── main.py                 → FastAPI app (inference)
├── Dockerfile              → Containerization
├── requirements.txt        → Dependencies
└── README.md
```

---

# ⚙️ Tech Stack

* **Language:** Python
* **Model:** XGBoost Classifier
* **Experiment Tracking:** MLflow
* **API:** FastAPI
* **Containerization:** Docker
* **Cloud Deployment:** Render
* **CI/CD:** GitHub Actions

---

# 🔄 ML Pipeline Workflow

### 1️⃣ Data Ingestion

* Load raw data from `data/raw.csv`

### 2️⃣ Data Preprocessing

* Handle missing values
* Encode categorical variables
* Feature engineering

### 3️⃣ Model Training

* Train XGBoost classifier
* Hyperparameter tuning using RandomizedSearchCV

### 4️⃣ Evaluation

* Metrics:

  * Accuracy
  * ROC-AUC

### 5️⃣ Experiment Tracking

* Log parameters, metrics, and models using MLflow

### 6️⃣ Model Saving

* Save best model in `model/` directory

---

# 🧠 Model Details

* Algorithm: **XGBoost Classifier**
* Objective: Binary Classification
* Key tuned parameters:

  * `n_estimators`
  * `max_depth`
  * `learning_rate`
  * `subsample`
  * `colsample_bytree`

---

# 🌐 Production API (Deployed on Render)

### 🔗 Base URL:

```
https://<your-actual-app-name>.onrender.com
```

### 📌 Endpoint:

```
POST /predict
```

### 📥 Sample Request:

```json
{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "Contract": "Month-to-month"
}
```

### 📤 Sample Response:

```json
{
  "prediction": "No Churn"
}
```

### 🧪 Test using curl:

```bash
curl -X POST "https://<your-actual-app-name>.onrender.com/predict" \
-H "Content-Type: application/json" \
-d '{"tenure":12,"MonthlyCharges":70.5,"Contract":"Month-to-month"}'
```

---

# 🧪 Local Development (Optional)

```bash
git clone <your-repo-link>
cd telecom-churn-mlops

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

python pipeline/training_pipeline.py
python main.py
```

---

# 📊 MLflow Tracking

```bash
mlflow ui
```

Access at:

```
http://localhost:5000
```

---

# 🐳 Docker Deployment

```bash
docker build -t churn-mlops .
docker run -p 8000:8000 churn-mlops
```

---

# 🔁 CI/CD Pipeline

Implemented using GitHub Actions:

* Trigger on push
* Install dependencies
* Run checks/tests
* Ensure pipeline integrity

Config:

```
.github/workflows/ci-cd.yml
```

---

# 📦 Project Highlights

* ✅ End-to-end ML pipeline
* ✅ XGBoost with hyperparameter tuning
* ✅ MLflow experiment tracking
* ✅ FastAPI deployment
* ✅ Docker containerization
* ✅ Cloud deployment on Render
* ✅ CI/CD integration

---

# 🧭 Future Improvements

* Add model monitoring
* Add data validation (Great Expectations)
* Implement feature store

---

# 👨‍💻 Author

Built to demonstrate **production-ready MLOps skills** for real-world machine learning systems.

---

# ⭐ Final Note

This project shows the ability to:

> **Build, track, deploy, and manage ML systems in a production environment.**

---

