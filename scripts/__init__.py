"""
This package contains scripts for:
- Processing research papers (PDF to Markdown, feature extraction)
- Training machine learning models
- Making predictions on research papers
"""

# Import key functions for easier access
from .parsing.preprocess import process_pdfs
from .parsing.features import process_directory
from .modeling.train_models import tune_svm, evaluate_model
from .modeling.train_neuralnet import save_metrics, save_confusion_matrix
from .predictions.predict import load_train_data

# Define an accessible package version
__version__ = "1.0.0"
