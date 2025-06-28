import time
import argparse
import os
import shutil
import tempfile
import numpy as np
from scripts import process_pdfs
from scripts import process_directory
from keras import models as mod
from sklearn.preprocessing import MinMaxScaler
from scripts import load_train_data

# Paths
MODEL_PATH = "models/neuralnet_model.h5"  # Neural Network Model Path
input_train_dir = "data/processed/reference"  # For scaling of new data


def load_model():
    """Loads the trained NN model."""
    if MODEL_PATH.endswith(".h5"):  # Keras NN Model
        model = mod.load_model(MODEL_PATH)
    else:
        raise ValueError("Unsupported model format")
    return model


def classify_paper(pdf_path, model):
    """Processes the input PDF and classifies it as publishable or not."""
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory

    try:
        # Step 1: Copies the PDF to temp directory
        temp_pdf_path = os.path.join(temp_dir, os.path.basename(pdf_path))
        shutil.copy(pdf_path, temp_pdf_path)

        # Step 1: Converts PDF to Markdown
        markdown_folder = os.path.join(temp_dir, "markdown")
        os.makedirs(markdown_folder, exist_ok=True)
        process_pdfs(temp_dir, markdown_folder)

        # Ensures all file handles are released
        time.sleep(2)

        # Step 2: Processes Markdown to NumPy
        npy_folder = os.path.join(temp_dir, "npy")
        os.makedirs(npy_folder, exist_ok=True)
        process_directory(markdown_folder, npy_folder)

        # Step 3: Retrieves the Numpy array of features
        npy_path = os.path.join(npy_folder, os.listdir(npy_folder)[0])
        feature_vector = np.load(npy_path)

        if feature_vector is None:
            print("Error: No valid feature vector extracted.")
        else:
            feature_vector = feature_vector.reshape(1, -1)  # Ensures correct shape
            X_train = load_train_data()
            sc = MinMaxScaler()
            _ = sc.fit_transform(X_train)
            feature_scaled = sc.transform(feature_vector)

            prediction = (model.predict(feature_scaled) >= 0.6).astype(int)
            print(model.predict(feature_scaled))
            label = "Publishable" if prediction[0] == 1 else "Not Publishable"

        return label

    finally:
        time.sleep(2)
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            print(
                f"Warning: Could not delete {temp_dir} immediately. Try deleting it manually."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classifies a research paper as publishable or not"
    )
    parser.add_argument(
        "pdf_path", type=str, help="Path to the research paper PDF file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print("Error: File not found.")
        exit(1)

    model = load_model()
    result = classify_paper(args.pdf_path, model)

    if result:
        print(f"Classification: {result}")
