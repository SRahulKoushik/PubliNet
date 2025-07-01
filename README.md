# PubliNet 📄

**Publishability Classifier for Research Papers**

---

## 🧐 Overview

PubliNet is an end-to-end machine learning pipeline for classifying research papers as "publishable" or "not publishable" based on their content. It automates the journey from raw PDF ingestion, through feature extraction, to prediction using multiple trained models (Neural Network, SVM, XGBoost). Designed for researchers, publishers, and developers interested in automating or analyzing the publishability of scientific manuscripts.

---

## ✨ Features

- 📥 **PDF to Markdown Conversion:** Uses [marker-pdf](https://github.com/allenai/marker) to convert research paper PDFs into structured Markdown.
- 🧬 **Feature Extraction:** Extracts a rich set of features from Markdown, including text statistics, section/keyword counts, sentiment, and BERT embeddings.
- 🤖 **Model Training:** Supports Neural Network, SVM, and XGBoost models with hyperparameter tuning and evaluation.
- 🏷️ **Prediction:** Classifies new papers and provides detailed metrics and confusion matrices.
- 🧩 **Extensible Pipeline:** Modular scripts for preprocessing, modeling, and prediction.

---

## 🗂️ Project Structure

```
PubliNet/
│
├── data/
│   ├── raw/           # Raw PDF files
│   ├── interim/       # Intermediate Markdown files
│   └── processed/     # Extracted features as .npy files
│
├── models/            # Trained model files (.h5, .pkl, .json)
├── results/
│   └── figures/       # Output plots (confusion matrix, metrics, etc.)
├── scripts/
│   ├── parsing/       # PDF-to-Markdown and feature extraction
│   ├── modeling/      # Model training scripts
│   └── predictions/   # Prediction and evaluation scripts
├── run.py             # Main entry point for classifying a paper
├── requirements.txt   # Python dependencies
└── LICENSE            # MIT License
```

---

## ⚡ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/PubliNet.git
   cd PubliNet
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data (first run only):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

---

## 🚦 Usage

### 1. Classify a Research Paper

```bash
python run.py path/to/your_paper.pdf
```

- The script will output whether the paper is "Publishable" or "Not Publishable".

### 2. Train Models

- **Neural Network:**  
  `python scripts/modeling/train_neuralnet.py`
- **SVM/XGBoost:**  
  `python scripts/modeling/train_models.py`

### 3. Feature Extraction

- **Convert PDFs to Markdown:**  
  `python scripts/parsing/preprocess.py`
- **Extract Features:**  
  `python scripts/parsing/features.py`

### 4. Batch Prediction & Evaluation

- `python scripts/predictions/predict.py`

---

## 📥 Input / 📤 Output

- **Input:** PDF files in `data/raw/`
- **Intermediate:** Markdown in `data/interim/`, features in `.npy` in `data/processed/`
- **Output:** Classification results, metrics, and plots in `results/figures/`

---

## 🏆 Model Performance

After extensive evaluation, the **Neural Network** model outperformed both SVM and XGBoost in terms of accuracy, precision, recall, and F1-score. All three models are available, but for best results, use the neural network (`models/neuralnet_model.h5`).

- 🤖 **Neural Network:** Best overall performance ✅
- 🦾 **SVM:** Good, but not as strong as NN
- 🌲 **XGBoost:** Competitive, but NN leads

See `results/figures/` for:
- `confusion_matrix.png`
- `metrics_table.png`
- `loss_plot.png`

---

## 🧠 Models

- `models/neuralnet_model.h5` (Keras Neural Network)
- `models/svm_model.pkl` (SVM)
- `models/xgboost_model.json` (XGBoost)

---

## 🛠️ Extending

- Add new feature extraction logic in `scripts/parsing/features.py`
- Add or swap models in `scripts/modeling/`
- Use your own data by placing PDFs in `data/raw/`

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [marker-pdf](https://github.com/allenai/marker) for PDF-to-Markdown conversion
- [HuggingFace Transformers](https://huggingface.co/transformers/) for BERT embeddings

---

## 👤 Contact

Maintained by **Sodisetty Rahul Koushik** and **Kesamreddy Yedukondala Yashwanth Reddy**
