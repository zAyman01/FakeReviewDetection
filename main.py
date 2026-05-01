import os
import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK resources
def setup_nltk():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for r in resources:
        try:
            nltk.data.find(r)
        except LookupError:
            nltk.download(r)

setup_nltk()

MODEL_NAMES = [
    "Logistic Regression",
    "Naive Bayes",
    "SVM",
    "Random Forest",
    "Deep Learning (LSTM/GRU)",
]

MODELS_DIR = "models"
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

class PredictionEngine:
    def __init__(self):
        self.vectorizer = self._load_joblib("tfidf_vectorizer.pkl")
        self.logreg = self._load_joblib("logreg_model.pkl")
        self.nb = self._load_joblib("nb_model.pkl")
        self.svm = self._load_joblib("svm_model.pkl")
        self.rf = self._load_joblib("rf_model.pkl")
        
        # Deep Learning
        self.dl_model = self._load_keras("deep_learning_model.keras")
        self.tokenizer = self._load_joblib("tokenizer.pkl")
        
        metadata_path = os.path.join(MODELS_DIR, "deep_learning_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.dl_metadata = json.load(f)
        else:
            self.dl_metadata = None

    def _load_joblib(self, filename):
        path = os.path.join(MODELS_DIR, filename)
        return joblib.load(path) if os.path.exists(path) else None

    def _load_keras(self, filename):
        path = os.path.join(MODELS_DIR, filename)
        return tf.keras.models.load_model(path) if os.path.exists(path) else None

    def predict(self, text):
        cleaned = clean_text(text)
        if not cleaned:
            return {name: "Invalid Input" for name in MODEL_NAMES}
        
        results = {}
        
        # TF-IDF based models
        if self.vectorizer:
            X_tfidf = self.vectorizer.transform([cleaned])
            results["Logistic Regression"] = self._format_pred(self.logreg.predict(X_tfidf)[0]) if self.logreg else "Model Missing"
            results["Naive Bayes"] = self._format_pred(self.nb.predict(X_tfidf)[0]) if self.nb else "Model Missing"
            results["SVM"] = self._format_pred(self.svm.predict(X_tfidf)[0]) if self.svm else "Model Missing"
            results["Random Forest"] = self._format_pred(self.rf.predict(X_tfidf)[0]) if self.rf else "Model Missing"
        else:
            for name in MODEL_NAMES[:4]: results[name] = "Vectorizer Missing"

        # Deep Learning
        if self.dl_model and self.tokenizer and self.dl_metadata:
            seq = self.tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=self.dl_metadata.get('max_len', 100), padding='post', truncating='post')
            prob = self.dl_model.predict(pad, verbose=0)[0][0]
            results["Deep Learning (LSTM/GRU)"] = self._format_pred(prob > 0.5)
        else:
            status = "Model Missing" if not self.dl_model else "Metadata Missing"
            results["Deep Learning (LSTM/GRU)"] = status
            
        return results

    def _format_pred(self, label):
        return "FAKE (Computer Generated)" if label == 1 else "REAL (Original)"

engine = None

def get_predictions(review_text: str):
    global engine
    if engine is None:
        try:
            engine = PredictionEngine()
        except Exception as e:
            return {name: f"Error: {str(e)}" for name in MODEL_NAMES}
    return engine.predict(review_text)


def on_predict(input_box, output_vars):
    review_text = input_box.get("1.0", tk.END).strip()
    if not review_text:
        messagebox.showwarning("Input Error", "Please enter a review text to analyze.")
        return
    predictions = get_predictions(review_text)
    for model_name, var in output_vars.items():
        var.set(predictions[model_name])


def on_clear(input_box, output_vars):
    input_box.delete("1.0", tk.END)
    for var in output_vars.values():
        var.set("Waiting for input...")


def build_gui(root):
    root.title("Fake Product Review Detection")
    root.geometry("800x600")

    style = ttk.Style()
    style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)

    # Header
    ttk.Label(frame, text="Fake Product Review Detection System", style="Header.TLabel").pack(pady=(0, 15))

    # Input Section
    ttk.Label(frame, text="Enter Review Text:").pack(anchor="w")
    input_box = ScrolledText(frame, height=8, font=("Helvetica", 10))
    input_box.pack(fill="both", expand=False, pady=(5, 15))

    # Action Buttons
    btn_frame = ttk.Frame(frame)
    btn_frame.pack(fill="x", pady=(0, 15))
    
    ttk.Button(btn_frame, text="Predict", command=lambda: on_predict(input_box, output_vars)).pack(side="right", padx=5)
    ttk.Button(btn_frame, text="Clear", command=lambda: on_clear(input_box, output_vars)).pack(side="right")

    # Results Section
    output_vars = {}
    result_box = ttk.LabelFrame(frame, text="Model Classifications", padding=15)
    result_box.pack(fill="both", expand=True)

    for row, model_name in enumerate(MODEL_NAMES):
        ttk.Label(result_box, text=f"{model_name}:", font=("Helvetica", 9, "bold")).grid(row=row, column=0, sticky="w", padx=(0, 15), pady=6)
        var = tk.StringVar(value="Waiting for input...")
        output_vars[model_name] = var
        ttk.Entry(result_box, textvariable=var, width=70, state="readonly").grid(row=row, column=1, sticky="ew", pady=6)
    
    result_box.columnconfigure(1, weight=1)

    # Status Bar
    status_bar = ttk.Label(root, text="Ready", relief="sunken", anchor="w", padding=(5, 2))
    status_bar.pack(side="bottom", fill="x")


def main():
    root = tk.Tk()
    build_gui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

