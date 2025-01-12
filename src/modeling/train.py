import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, make_scorer
from keras import Sequential
from keras import layers
from sklearn.utils import compute_class_weight
import pandas as pd
from keras import regularizers
from scikeras.wrappers import KerasClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import joblib

# Define input directory
input_dir = "data/processed/megaset"

# Load data function (update with actual path)
def load_data():
    X = []
    y = []

    for filename in os.listdir(input_dir):
            data = np.array(np.load(os.path.join(input_dir, filename), allow_pickle=True))

            X.append(data)  # All columns except the last one (features)

    X = np.array(X)
    y = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
    # return train_test_split(X, y, test_size=0.3, random_state=42)
    return X

# Function to evaluate models and return all metrics
def evaluate_model_xgboost(model, X_test, y_test, model_name):
    X_test = xgb.DMatrix(X_test)
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

# Custom F1 Metric
def f1_metric(y_true, y_pred):
    y_pred = np.round(y_pred)  # Threshold at 0.5
    return f1_score(y_true, y_pred)

# Create Neural Network Function
def create_nn(units1=128, units2=64, dropout1=0.3, dropout2=0.3, l1_reg=0.01, l2_reg=0.01):
    model = Sequential([
        layers.Dense(units=units1, activation="relu", 
                     kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
        layers.Dropout(dropout1),
        layers.BatchNormalization(),
        layers.Dense(units=units2, activation="relu", 
                     kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)),
        layers.Dropout(dropout2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Simulated Data (Replace with actual data loading)
X_train = load_data() # Example features
y_train = np.array([0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]) # Example labels

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Calculate Class Weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Wrap Model for Grid Search
model = KerasClassifier(
    build_fn=create_nn,
    verbose=0,
    units1=128,  # Default values for hyperparameters
    units2=64,
    dropout1=0.3,
    dropout2=0.3,
    l1_reg=0.01,
    l2_reg=0.01
)

# Hyperparameter Grid
param_grid = {
    'units1': [64, 128],
    'units2': [32, 64],
    'dropout1': [0.2, 0.3],
    'dropout2': [0.2, 0.3],
    'l1_reg': [0.01, 0.001],
    'l2_reg': [0.01, 0.001]
}

# K-Fold Cross-Validation
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

# Perform Grid Search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold,
    scoring=make_scorer(f1_metric)
)
grid_result = grid.fit(X_train, y_train, class_weight=class_weights_dict)

# Output Best Results
print("Best F1-Score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Train the Best Model on the Entire Dataset
best_params = grid_result.best_params_
best_model = create_nn(**best_params)
best_model.fit(X_train, y_train, epochs=10, batch_size=32, class_weight=class_weights_dict)

# XGBoost model without GridSearchCV
def train_xgboost(X_train, y_train):
    # Ensure X_train has non-zero features
    if X_train.shape[1] == 0:
        raise ValueError("X_train has no features. Check your data loading process.")

    # Ensure X_train is a 2D array and y_train is 1D
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)  # Make X_train 2D if it's 1D

    # Convert to np.array if using pandas DataFrame
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Check for NaN or infinite values in the data
    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
        raise ValueError("Input data contains NaN values.")
    if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
        raise ValueError("Input data contains infinite values.")

    # Create XGBoost DMatrix for training
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # # Set parameters for XGBoost (default values used)
    # params = {
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'logloss',
    #     'max_depth': 5,
    #     'eta': 0.1,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'learning_rate': 0.05,
    #     'n_jobs': 8
    # }

    # # Train the model using XGBoost
    # xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    # print("Training completed for XGBoost.")
    # return xgb_model

    # Instantiate the XGBoost classifier
    xgb_model = xgb.XGBClassifier()
    
    pipeline = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ('pca', PCA()), 
    ('model', xgb_model)
    ])
    
    # # Define the parameter grid for GridSearchCV
    # param_grid = {
    #     'pca__n_components': [5, 10, 15, 20, 25, 30],
    #     'model__max_depth': [2, 3, 5, 7, 10],
    #     'model__n_estimators': [10, 100, 500],
    #     'learning_rate': [0.01, 0.1, 0.2],    # Step size shrinkage
    #     'subsample': [0.7, 0.8, 1.0],         # Fraction of samples used per tree
    #     'colsample_bytree': [0.7, 0.8, 1.0],  # Fraction of features used per tree
    #     'gamma': [0, 1, 5],                   # Minimum loss reduction for split
    #     'scale_pos_weight': [1, 2, 3],        # To balance class imbalance
    # }
    
    param_grid = {
    'model__max_depth': [2, 3, 5, 7, 10],     # Focus on medium complexity for better generalization
    'model__n_estimators': [10, 100, 200, 300], # Prioritize fewer estimators to avoid long training times
    'model__learning_rate': [0.01, 0.05, 0.1],  # Add learning rate for fine-grained tuning
    'model__subsample': [0.7, 0.8, 1.0],   # For sampling training instances
    'model__colsample_bytree': [0.7, 0.8, 1.0], # For sampling features per tree
    'model__gamma': [0, 0.1, 0.2, 0.3],   # Control overfitting by specifying minimum loss reduction for split
    'model__scale_pos_weight': [1, 2, 3],  # Adjust for class imbalance (ratio of negative to positive class)
    }

    # Perform GridSearchCV
    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring='f1',  # Metric to optimize
        cv=5,          # 3-fold cross-validation
        n_jobs=-1      # Run parallel jobs
    )

    # Fit the grid search to the training data
    grid.fit(X_train, y_train)

    print("Best parameters found for XGBoost:", grid.best_params_)
    print("Best F1 Score from GridSearch:", grid.best_score_)

    # Return the best estimator
    return grid.best_estimator_


# SVM model with hyperparameter tuning
def tune_svm(X_train, y_train):
    param_grid = {
        'svc__C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter for SVM
        'svc__kernel': ['linear', 'rbf', 'poly'],  # Types of kernels to try
        'svc__gamma': ['scale', 'auto', 0.1, 0.5],  # Influence of individual training examples
        'svc__degree': [2, 3, 4],  # Only for poly kernel, the degree of the polynomial
        'svc__class_weight': [None, 'balanced'],  # Adjust weights for class imbalance
        'svc__tol': [1e-3, 1e-4],  # Stopping criteria tolerance (controls optimization precision)
        'svc__max_iter': [-1, 1000, 5000]  # Limit maximum number of iterations to avoid overfitting
    }
    
    # Set up the pipeline with scaling, PCA, and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA()),  # Optionally use PCA for dimensionality reduction
        ('svc', SVC())  # SVM model
    ])

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',  # Can switch to 'f1' or 'roc_auc' based on your evaluation metric
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,  # Use all available cores for faster computation
        verbose=2,  # Verbose output to track progress
        refit=True  # Refit the model with the best parameters found
    )

    # Ensure X_train and X_test are NumPy arrays before passing to the SVM
    X_train = np.array(X_train)

    # Perform grid search to find the best hyperparameters
    grid_search.fit(X_train, y_train)

    # Print out the best parameters and the corresponding score
    print("Best Parameters Found: ", grid_search.best_params_)
    print("Best Cross-validation Accuracy: ", grid_search.best_score_)

    # Return the best model from GridSearchCV
    return grid_search.best_estimator_

# Logistic Regression model with hyperparameter tuning
def train_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000]
    }
    lr_model = LogisticRegression()
    grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters for Logistic Regression:", grid_search.best_params_)
    return grid_search.best_estimator_

# Main function
if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = load_data()

        # # Handle class imbalance with class weights (for models that need it)
        # class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        # class_weight_dict = dict(enumerate(class_weights))

        # # Train and evaluate XGBoost without GridSearchCV
        # xgb_model = train_xgboost(X_train, y_train)
        # xgb_metrics = evaluate_model_xgboost(xgb_model, X_test, y_test, "XGBoost")
        # xgb_model.save_model("xgboost_model.json")

        # # Train and evaluate SVM
        # svm_model = tune_svm(X_train, y_train)
        # svm_metrics = evaluate_model(svm_model, X_test, y_test, "SVM")
        # joblib.dump(svm_model, "svm_model.pkl")

        # # Train and evaluate Logistic Regression
        # lr_model = train_logistic_regression(X_train, y_train)
        # lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

        # Train and evaluate Neural Network
        nn_model = create_nn(X_train, y_train, X_test, y_test)
        nn_metrics = evaluate_model(nn_model, X_test, y_test, "Neural Network")

        # Compare all metrics
        # models = ['XGBoost', 'SVM', 'Logistic Regression', 'Neural Network']
        # all_metrics = {
        #     'Accuracy': [xgb_metrics[0], svm_metrics[0], lr_metrics[0], nn_metrics[0]],
        #     'Precision': [xgb_metrics[1], svm_metrics[1], lr_metrics[1], nn_metrics[1]],
        #     'Recall': [xgb_metrics[2], svm_metrics[2], lr_metrics[2], nn_metrics[2]],
        #     'F1 Score': [xgb_metrics[3], svm_metrics[3], lr_metrics[3], nn_metrics[3]],
        # }
        
        models = ['Neural Network']
        all_metrics = {
             'Accuracy': [nn_metrics[0]],
             'Precision': [nn_metrics[1]],
             'Recall': [nn_metrics[2]],
             'F1 Score': [nn_metrics[3]],
        }

        # Convert metrics to a DataFrame for easy comparison
        metrics_df = pd.DataFrame(all_metrics, index=models)
        print("\nComparison of Model Metrics:")
        print(metrics_df)

        # Find the model with the highest F1 score (or another metric of your choice)
        best_model_index = metrics_df['F1 Score'].idxmax()
        print(f"\nThe best model based on F1 Score is: {best_model_index}")

    except Exception as e:
        print(f"Error: {e}")

