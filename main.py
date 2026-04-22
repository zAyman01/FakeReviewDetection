import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


MODEL_NAMES = [
    "Logistic Regression",
    "Naive Bayes",
    "SVM",
    "Random Forest",
    "Deep Learning (LSTM/GRU)",
]


def get_predictions(review_text: str):
    if not review_text:
        return {name: "" for name in MODEL_NAMES}
    return {name: "Pending (to be implemented)" for name in MODEL_NAMES}


def on_predict(input_box, output_vars):
    review_text = input_box.get("1.0", tk.END).strip()
    predictions = get_predictions(review_text)
    for model_name, var in output_vars.items():
        var.set(predictions[model_name])


def build_gui(root):
    root.title("Fake Product Review Detection")
    root.geometry("760x420")

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Review Text").pack(anchor="w")
    input_box = ScrolledText(frame, height=7)
    input_box.pack(fill="both", expand=False, pady=(6, 12))

    output_vars = {}
    result_box = ttk.LabelFrame(frame, text="Model Predictions", padding=10)
    result_box.pack(fill="x")

    for row, model_name in enumerate(MODEL_NAMES):
        ttk.Label(result_box, text=f"{model_name}:").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        var = tk.StringVar(value="")
        output_vars[model_name] = var
        ttk.Entry(result_box, textvariable=var, width=60, state="readonly").grid(row=row, column=1, sticky="w", pady=4)

    ttk.Button(frame, text="Predict", command=lambda: on_predict(input_box, output_vars)).pack(anchor="e", pady=(12, 0))


def main():
    root = tk.Tk()
    build_gui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

