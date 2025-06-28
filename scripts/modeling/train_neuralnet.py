import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from keras import Sequential
from keras import layers, regularizers
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
)


def load_data(input_dir):
    """
    Loads and returns data from the specified directory.

    Parameters:
    input_dir (str): Path to the directory containing NumPy files.

    Returns:
    np.ndarray: Array containing loaded data.
    """
    X = []
    for filename in os.listdir(input_dir):
        data = np.array(np.load(os.path.join(input_dir, filename), allow_pickle=True))
        X.append(data)
    return np.array(X)


def preprocess_data(X, y, test_size=0.15):
    """
    Preprocesses data by performing train-test split, scaling, and computing class weights.

    Parameters:
    X (np.ndarray): Feature dataset.
    y (np.ndarray): Target labels.
    test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.15.

    Returns:
    tuple: Scaled training and test datasets, labels, and class weights.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=0
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    return X_train, X_test, y_train, y_test, class_weights_dict, class_weights


def build_model(optimizer_set="adam", input_shape=None, weights=None):
    classifier = Sequential()

    classifier.add(
        layers.Dense(
            units=128,
            activation="relu",
            input_shape=(input_shape,),
            kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.01),
        )
    )
    for units in [64, 32, 16]:
        classifier.add(
            layers.Dense(
                units=units,
                activation="relu",
                kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.01),
            )
        )
    classifier.add(layers.Dense(units=1, activation="sigmoid"))

    classifier.compile(
        loss="binary_crossentropy",
        optimizer=optimizer_set,
        metrics=["accuracy"],
    )
    return classifier


def tune_hyperparameters(X_train, y_train):
    parameters = {
        "batch_size": [16, 32, 64, 128],
        "epochs": [100, 150, 200],
        "optimizer": ["adam", "rmsprop", "sgd"],
    }
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=0)
    random_search = RandomizedSearchCV(
        estimator=KerasClassifier(
            build_fn=lambda: build_model(input_shape=X_train.shape[1])
        ),
        param_distributions=parameters,
        n_iter=25,
        scoring="accuracy",
        cv=skf,
        random_state=0,
    )
    random_search.fit(X_train, y_train, verbose=0)
    return random_search.best_params_


def determine_threshold(model, X_test, y_test):
    thresholds = np.linspace(0.1, 0.9, 30)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (model.predict(X_test) >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, average="binary")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def save_metrics(y_test, y_pred, file_path="results/figures/metrics_table.png"):
    """
    Generates and saves evaluation metrics (F1-score, accuracy, precision, recall) as a figure.

    Parameters:
    y_test (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.
    file_path (str): Path to save the metrics table.
    """
    # Computes the metrics
    metrics = [
        ("Accuracy", accuracy_score(y_test, y_pred)),
        ("Precision", precision_score(y_test, y_pred)),
        ("Recall", recall_score(y_test, y_pred)),
        ("F1 Score", f1_score(y_test, y_pred)),
    ]

    # Creates a figure
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis("tight")
    ax.axis("off")

    # Creates the table
    table = ax.table(
        cellText=[
            [name, f"{value:.2f}"] for name, value in metrics
        ],  # Formatted to 2 decimal places
        colLabels=["Metric", "Score"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])

    # Saves the figure
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_confusion_matrix(
    y_test, y_pred, file_path="results/figures/confusion_matrix.png"
):
    """
    Generates and saves the confusion matrix as an image file.

    Parameters:
    y_test (np.ndarray): True labels.
    y_pred (np.ndarray): Predicted labels.
    file_path (str): Path to save the confusion matrix image.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(file_path)
    plt.close()


def main():
    input_dir = "data/processed/reference"
    X = load_data(input_dir)
    y = np.array([0] * 5 + [1] * 20 + [0] * 5)
    X_train, X_test, y_train, y_test, class_weights_dict, class_weights = (
        preprocess_data(X, y)
    )

    best_params = tune_hyperparameters(X_train, y_train)
    print("Best Parameters:", best_params)

    best_model = build_model(best_params["optimizer"], X_train.shape[1], class_weights)
    history = best_model.fit(
        X_train,
        y_train,
        batch_size=best_params["batch_size"],
        epochs=best_params["epochs"],
        class_weight=class_weights_dict,
        verbose=0,
    )

    loss_plot_path = "results/figures/loss_plot.png"
    plt.plot(history.history["loss"], label="NN Model")
    plt.title("Loss Function Over Epochs")
    plt.ylabel("Loss Value")
    plt.xlabel("Epochs")
    plt.legend(loc="upper right")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss function plot saved to {loss_plot_path}")

    best_threshold = determine_threshold(best_model, X_test, y_test)
    print(f"Best Threshold: {best_threshold}")

    y_pred = (best_model.predict(X_test) >= best_threshold).astype(int)
    f1 = f1_score(y_test, y_pred, average="binary")
    print(f"Best Classifier F1 Score: {f1}")

    model_save_path = "models/neuralnet_model.h5"
    best_model.save(model_save_path)
    print(f"Best model saved to {model_save_path}")


if __name__ == "__main__":
    main()
