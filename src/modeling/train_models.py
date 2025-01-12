import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Define input directory
input_dir = "data/processed/megaset"

# Function to load data
def load_data():
    X = []
    for filename in os.listdir(input_dir):
        data = np.load(os.path.join(input_dir, filename), allow_pickle=True)
        X.append(data)

    X = np.array(X)
    y = np.array([
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
    ])
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test)
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    cm = confusion_matrix(y_test, y_pred_binary)

    print(f"\n{model_name} Model Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, precision, recall, f1, cm

# XGBoost model with hyperparameter tuning
def train_xgboost(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("xgb", xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")),
    ])

    param_grid = {
        "xgb__max_depth": [3, 5, 7],
        "xgb__n_estimators": [50, 100, 200],
        "xgb__learning_rate": [0.01, 0.05, 0.1],
        "xgb__subsample": [0.8, 1.0],
        "xgb__colsample_bytree": [0.8, 1.0],
        "xgb__gamma": [0, 0.1, 0.2],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="accuracy",
        cv=7,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    print("Best parameters for XGBoost:", grid.best_params_)
    print("Best F1 Score for XGBoost:", grid.best_score_)

    return grid.best_estimator_

# SVM model with hyperparameter tuning
def tune_svm(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("svc", SVC(probability=True)),
    ])

    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["linear", "rbf"],
        "svc__gamma": ["scale", 0.1, 0.5],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="accuracy",
        cv=7,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters for SVM:", grid_search.best_params_)
    print("Best F1 Score for SVM:", grid_search.best_score_)

    return grid_search.best_estimator_

# Main function
if __name__ == "__main__":
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data()

        # Scale test data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train and evaluate XGBoost
        print("\nTraining XGBoost...")
        xgboost_model = train_xgboost(X_train, y_train)
        evaluate_model(xgboost_model, X_test, y_test, "XGBoost")

        # Train and evaluate SVM
        print("\nTraining SVM...")
        svm_model = tune_svm(X_train, y_train)
        evaluate_model(svm_model, X_test, y_test, "SVM")

    except Exception as e:
        print(f"Error: {e}")

