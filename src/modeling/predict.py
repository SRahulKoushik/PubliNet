import numpy as np
import joblib
import xgboost as xgb
import os
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)
from keras import models as mod
from sklearn.preprocessing import MinMaxScaler


# Load the saved models
xgb_model = xgb.Booster()
xgb_model.load_model("xgboost_model.json")

svm_model = joblib.load("svm_model.pkl")

# Define input directory
input_dir = "data/processed/dataset"


# Load data function (update with actual path)
def load_data():
    X = []

    for filename in os.listdir(input_dir):
        data = np.array(np.load(os.path.join(input_dir, filename), allow_pickle=True))

        X.append(data)  # All columns except the last one (features)

    X = np.array(X)

    return X


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


# Example usage
if __name__ == "__main__":
    # Example data
    X_test = load_data()  # Replace with actual test data
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
            1
        ]
    )

    # # Predict and evaluate with XGBoost
    y_pred_xgb, _ = predict_xgboost(xgb_model, X_test)
    evaluate_predictions(y_test, y_pred_xgb, "XGBoost")

    # Predict and evaluate with SVM
    y_pred_svm = predict_svm(svm_model, X_test)
    evaluate_predictions(y_test, y_pred_svm, "SVM")
    
    model_path = "nn_model_6.h5"  # Path to the saved model
    model = mod.load_model(model_path)
    sc = MinMaxScaler((0, 1))
    X_test_scaled = sc.fit_transform(X_test)
    
    y_pred_nn = (model.predict(X_test_scaled) > 0.52).astype(int)
    evaluate_predictions(y_test, y_pred_nn, "NN")
