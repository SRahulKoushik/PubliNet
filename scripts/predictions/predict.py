import sys
import os
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
from keras import models as mod
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts import save_confusion_matrix, save_metrics


# Load the saved models
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_model.json")

svm_model = joblib.load("models/svm_model.pkl")

# Define input directory
input_dir = "data/processed/papers"
input_train_dir = "data/processed/reference"


# Load data function (update with actual path)
def load_data():
    X = []

    for filename in os.listdir(input_dir):
        data = np.array(np.load(os.path.join(input_dir, filename), allow_pickle=True))

        X.append(data)  # All columns except the last one (features)

    X = np.array(X)

    return X


def load_train_data():
    """
    Loads and returns data from the specified directory.

    Parameters:
    input_dir (str): Path to the directory containing NumPy files.

    Returns:
    np.ndarray: Array containing loaded data.
    """
    X = []
    for filename in os.listdir(input_train_dir):
        data = np.array(
            np.load(os.path.join(input_train_dir, filename), allow_pickle=True)
        )
        X.append(data)
    return np.array(X)


# Prediction functions
def predict_xgboost(model, X):
    dmatrix = xgb.DMatrix(X)
    y_pred_prob = model.predict(dmatrix)
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    return y_pred_binary, y_pred_prob


def predict_svm(model, X):
    y_pred_binary = model.predict(X)
    return y_pred_binary


def evaluate_predictions(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Model Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    save_confusion_matrix(y_true, y_pred)
    save_metrics(y_true, y_pred)


# Prediction
if __name__ == "__main__":
    # Example data
    X_train = load_train_data()
    X_test = load_data()  # Test data
    y_test = np.array(
        [
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
        ]
    )

    # # Predict and evaluate with XGBoost
    y_pred_xgb, _ = predict_xgboost(xgb_model, X_test)
    evaluate_predictions(y_test, y_pred_xgb, "XGBoost")

    # Predict and evaluate with SVM
    y_pred_svm = predict_svm(svm_model, X_test)
    evaluate_predictions(y_test, y_pred_svm, "SVM")

    model_path = "models/neuralnet_model.h5"  # Path to the saved model
    model = mod.load_model(model_path)
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    y_pred_nn = (model.predict(X_test_scaled) >= 0.6).astype(int)
    evaluate_predictions(y_test, y_pred_nn, "NN")
