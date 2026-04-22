# Fake Product Review Detection using Natural Language Processing

## Project Overview

This project aims to develop a Natural Language Processing (NLP) system capable of detecting whether a product review is genuine or fabricated. Fake reviews are widely used to influence purchasing decisions, and identifying them is important for maintaining trust in online platforms.

The system analyzes textual data and applies multiple machine learning and deep learning models to classify reviews as either **fake** or **real**.

---

## Objectives

* Apply NLP techniques to analyze textual reviews
* Perform preprocessing to clean and standardize input data
* Train and evaluate five different classification models
* Compare model performance
* Provide a simple graphical user interface (GUI) for user interaction

---

## Models

The project includes five different models, each implemented separately:

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest
* LSTM / GRU (Deep Learning)

Each model processes the same input and produces an independent prediction.

---

## System Workflow

1. **Input**
   The user enters a product review through the graphical interface.

2. **Preprocessing**
   The text will be cleaned and prepared using standard NLP techniques, including:

   * Lowercasing
   * Removing punctuation and stopwords
   * Tokenization
   * Optional stemming or lemmatization

3. **Feature Extraction**
   Text data will be converted into numerical representations:

   * TF-IDF for traditional machine learning models
   * Word embeddings for deep learning models

4. **Prediction**
   Each of the five models analyzes the processed input and generates a classification result.

5. **Output**
   The system displays the predictions from all models in the interface.

---

## User Interface

The project includes a simple graphical user interface that allows the user to:

* Enter a review
* Trigger processing through a button
* View predictions from all five models

---

## Dataset

The project uses a dataset of product reviews containing:

* Review text
* Corresponding labels (fake or real)

This dataset is used for training and evaluating all models.

---

## Project Structure

```text
FakeReviewDetection/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── .gitkeep
└── notebooks/
    ├── logistic_regression.ipynb
    ├── naive_bayes.ipynb
    ├── svm.ipynb
    ├── random_forest.ipynb
    └── lstm_gru.ipynb
```

---

## Current Status

* Initial project structure has been created
* GUI placeholder is implemented
* Model notebooks are prepared for development

Planned work includes:

* Implementing preprocessing pipeline
* Training and evaluating models
* Integrating models with the GUI

---

## How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

---

## Future Work

* Improve model performance using advanced preprocessing techniques
* Add evaluation metrics such as accuracy, precision, recall, and F1-score
* Enhance the graphical interface
* Deploy the system as a web-based application

---

## Team Responsibility

Each team member is responsible for implementing and evaluating one of the five models, in accordance with the project requirements.
