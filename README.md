# Fake Product Review Detection System

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/sklearn-1.3%2B-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## Project Overview
This project is a comprehensive Natural Language Processing (NLP) solution designed to distinguish between **Real (Original - OR)** and **Fake (Computer-Generated - CG)** product reviews. By leveraging a variety of Machine Learning and Deep Learning architectures, the system identifies the subtle "fingerprints" of synthetic text to maintain trust in digital marketplaces.

## Key Features
- **Multi-Model Ensemble:** Implements five distinct algorithms to ensure robust detection.
- **Unified Pipeline:** A standardized workflow from raw data ingestion to model persistence.
- **Modern GUI:** A user-friendly Tkinter-based interface for real-time review analysis.
- **Interactive Notebooks:** Documented exploratory data analysis and model tuning.
- **Automated Visualization:** Every notebook generates performance charts (Confusion Matrices, ROC Curves) saved directly to the project.

## Project Structure
```text
FakeReviewDetection/
├── main.py                 # Main GUI Application
├── test_models.py          # Backend connectivity test script
├── requirements.txt        # Project dependencies
├── data/
│   ├── raw/                # Original dataset (fake_reviews.csv)
│   ├── preprocessed/       # Cleaned text data
│   └── processed/          # TF-IDF features and labels
├── notebooks/              # Standardized Jupyter Notebooks
│   ├── preprocessing_eda.ipynb
│   ├── tfidf.ipynb
│   ├── logistic_regression.ipynb
│   ├── naive_bayes.ipynb
│   ├── svm.ipynb
│   ├── random_forest.ipynb
│   └── lstm_gru.ipynb
├── models/                 # Trained model artifacts (.pkl, .keras, .json)
└── figures/                # Exported visualizations and evaluation charts
```

## Algorithms & Performance
The system compares several approaches to identify the most effective detection method:

| Model | Feature Extraction | Accuracy |
| :--- | :--- | :--- |
| **Support Vector Machine (SVM)** | TF-IDF (Unigrams-Trigrams) | **91.26%** |
| **Logistic Regression** | TF-IDF (Unigrams-Trigrams) | 91.06% |
| **Deep Learning (LSTM/GRU)** | Word Embeddings (Keras) | 90.23% |
| **Random Forest** | TF-IDF (Unigrams-Trigrams) | 89.17% |
| **Complement Naive Bayes** | TF-IDF (Unigrams-Trigrams) | 88.52% |

## Installation & Setup

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/zAyman01/FakeReviewDetection.git
   cd FakeReviewDetection
   ```

2. **Set up a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

### Running the GUI
Launch the interactive detection tool:
```powershell
python main.py
```

### Running the Notebooks
To retrain models or explore the data:
1. Open Jupyter Lab/Notebook.
2. Run notebooks in order: `preprocessing_eda.ipynb` -> `tfidf.ipynb` -> Model Notebooks.

### Verification
Run the automated test script to verify all backend components:
```powershell
python test_models.py
```

## Visualizations
Detailed performance metrics are available in the `figures/` directory, including:
- **Class Distributions:** Understanding the balance between OR and CG reviews.
- **Confusion Matrices:** Visualizing True Positives vs. False Positives for every model.
- **Feature Importance:** Identifying the top words that trigger fake review detection.
- **Training History:** Loss and Accuracy curves for the Deep Learning model.

## License
This project is developed for educational and research purposes in the field of NLP and Fake Content Detection.
