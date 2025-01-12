import numpy as np  # linear algebra
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras import layers
from keras import regularizers
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, matthews_corrcoef, make_scorer

# Define input directory
input_dir = "data/processed/megaset"
X = []

# Load data function (update with actual path)
for filename in os.listdir(input_dir):
    data = np.array(np.load(os.path.join(input_dir, filename), allow_pickle=True))
    X.append(data)  # All columns except the last one (features)

X = np.array(X)
y = np.array(
    [
        0,
        0,
        0,
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
        0,
        0,
        0,
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reshape target arrays if necessary
if len(y_train.shape) == 1:
    y_train = y_train.reshape(-1, 1)
if len(y_test.shape) == 1:
    y_test = y_test.reshape(-1, 1)

# Scaling data
sc = MinMaxScaler((0, 1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# # Apply SMOTE to balance the classes
# smote = SMOTE(sampling_strategy="auto", k_neighbors=3, random_state=0)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Compute class weights
# class_weights = compute_class_weight(
#     "balanced", classes=np.unique(y_train_resampled), y=y_train_resampled
# )
# class_weight_dict = dict(zip(np.unique(y_train_resampled), class_weights))

# classifier = Sequential()

# # Adding the input layer and the first hidden layer
# classifier.add(layers.Dense(units=128, activation="relu", kernel_regularizer=regularizers.l2(0.01)))

# # Adding the second hidden layer
# classifier.add(layers.Dense(units=64, activation="relu", kernel_regularizer=regularizers.l2(0.01)))

# classifier.add(layers.Dense(units=32, activation="relu", kernel_regularizer=regularizers.l2(0.01)))

# # Adding the output layer
# classifier.add(layers.Dense(units=1, activation="sigmoid"))

# # Compiling the ANN
# classifier.compile(
#     loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "precision", "recall"]
# )

# History = classifier.fit(x=X_train, y=y_train, batch_size=128, epochs=150, verbose=0)

# plt.plot(History.history["loss"])
# plt.title("Loss Function Over Epochs")
# plt.ylabel("BCE value")
# plt.xlabel("No. epoch")
# plt.show()

# # Define custom MCC scorer
# mcc_score = make_scorer(matthews_corrcoef)


def buildModel(optimizer_set="adam"):
    classifier = Sequential()

    # Input layer + first hidden layer with L1 + L2 regularization + Batch Normalization + Dropout
    classifier.add(
        layers.Dense(
            units=128,
            activation="relu",
            input_shape=(X_train.shape[1],),
            kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.01),
        )
    )  # L1 + L2 regularization
    # classifier.add(layers.BatchNormalization())  # Batch Normalization
    # classifier.add(layers.Dropout(0.2))  # Dropout to prevent overfitting

    # Second hidden layer with L1 + L2 regularization + Batch Normalization + Dropout
    classifier.add(
        layers.Dense(
            units=64,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.01),
        )
    )  # L1 + L2 regularization
    # classifier.add(layers.BatchNormalization())  # Batch Normalization
    # classifier.add(layers.Dropout(0.19))  # Dropout

    # Third hidden layer with L1 + L2 regularization + Batch Normalization + Dropout
    classifier.add(
        layers.Dense(
            units=32,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.01),
        )
    )  # L1 + L2 regularization
    # classifier.add(layers.BatchNormalization())  # Batch Normalization
    # classifier.add(layers.Dropout(0.18))  # Dropout

    # Fourth hidden layer with L1 + L2 regularization + Batch Normalization + Dropout
    classifier.add(
        layers.Dense(
            units=16,
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.01),
        )
    )  # L1 + L2 regularization
    # classifier.add(layers.BatchNormalization())  # Batch Normalization
    # classifier.add(layers.Dropout(0.18))  # Dropout

    #  # Fifth hidden layer with L1 + L2 regularization + Batch Normalization + Dropout
    # classifier.add(
    #     layers.Dense(
    #         units=8,
    #         activation="relu",
    #         kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.01),
    #     )
    # )  # L1 + L2 regularization
    # # classifier.add(layers.BatchNormalization())  # Batch Normalization
    # # classifier.add(layers.Dropout(0.18))  # Dropout

    # Output layer
    classifier.add(
        layers.Dense(units=1, activation="sigmoid")
    )  # Output layer for binary classification

    # Compile the model with the specified optimizer and loss function
    classifier.compile(
        loss="binary_crossentropy",
        optimizer=optimizer_set,
        metrics=["accuracy"],
    )

    return classifier


# What hyperparameter we want to play with
parameters = {
    "batch_size": [16, 32, 64, 128],
    "epochs": [100, 150, 200],
    "optimizer": ["adam", "rmsprop", "sgd"],
}
grid_search = GridSearchCV(
    estimator=KerasClassifier(build_fn=buildModel),
    param_grid=parameters,
    # scoring=mcc_score,
    scoring="accuracy",
    cv=7,
)
grid_search = grid_search.fit(X_train, y_train, verbose=0)

best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters: " + str(best_parameters))

# Rebuild model with best parameters
bestClassifier = buildModel(optimizer_set=best_parameters["optimizer"])

# Train the best model
HistoryBest = bestClassifier.fit(
    X_train,
    y_train,
    batch_size=best_parameters["batch_size"],
    epochs=best_parameters["epochs"],
    verbose=0,
)

# Plot loss history
plt.plot(HistoryBest.history["loss"], label="Best Model")
plt.title("Loss Function Over Epochs")
plt.ylabel("BCE value")
plt.xlabel("Epochs")
plt.legend(loc="upper right")
plt.show()

# Make predictions on the test set
y_pred = (bestClassifier.predict(X_test) > 0.5).astype(int)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average="binary")
print(f"Best Classifier F1 Score: {f1}")

# Save the best model
model_save_path = "nn_model_6.h5"  # Specify the file path to save the model
bestClassifier.save(model_save_path)
print(f"Best model saved to {model_save_path}")
