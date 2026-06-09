"""
Handwritten Digit Recognition GUI
Starter portfolio project.

This application allows you to draw a handwritten digit and then uses a
trained machine learning model to recognize the digit from 0 to 9.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

MODEL_PATH = Path(__file__).with_name("digit_gui_model.joblib")
CANVAS_SIZE = 280
IMAGE_SIZE = 8
PEN_WIDTH = 24
BACKGROUND_COLOR = "black"
DRAW_COLOR = "white"


from sklearn.neural_network import MLPClassifier


def load_or_train_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        stratify=digits.target,
        random_state=42,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        early_stopping=True,
        random_state=42,
        max_iter=500,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    grayscale = image.convert("L")
    inverted = ImageOps.invert(grayscale)
    thresholded = inverted.point(lambda px: 255 if px > 50 else 0)

    array = np.asarray(thresholded, dtype=np.uint8)
    non_empty = np.argwhere(array)
    if non_empty.size == 0:
        cropped = thresholded
    else:
        min_y, min_x = non_empty.min(axis=0)
        max_y, max_x = non_empty.max(axis=0)
        cropped = thresholded.crop((min_x, min_y, max_x + 1, max_y + 1))

    size = max(cropped.width, cropped.height)
    padded = Image.new("L", (size + 20, size + 20), color=0)
    offset = ((padded.width - cropped.width) // 2, (padded.height - cropped.height) // 2)
    padded.paste(cropped, offset)

    resized = padded.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    final = ImageOps.invert(resized)
    array = np.asarray(final, dtype=np.float32)
    array = 16.0 * array / 255.0
    return array.reshape(1, -1)


class DigitRecognitionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition")
        self.model = load_or_train_model()

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=BACKGROUND_COLOR,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", lambda _: None)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw_image = ImageDraw.Draw(self.image)

        self.predict_button = ttk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, sticky="ew", padx=10)

        self.clear_button = ttk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=1, sticky="ew", padx=10)

        self.result_label = ttk.Label(
            self.root,
            text="Draw a digit between 0 and 9, then click Predict.",
            font=("Segoe UI", 12),
            wraplength=CANVAS_SIZE,
            justify="center",
        )
        self.result_label.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        self.extra_label = ttk.Label(
            self.root,
            text="Model trained on sklearn digits dataset (8x8 images) with MLP preprocessing.",
            font=("Segoe UI", 9),
            foreground="gray",
        )
        self.extra_label.grid(row=3, column=0, columnspan=2, pady=(0, 10))

    def draw(self, event: tk.Event):
        x1 = event.x - PEN_WIDTH // 2
        y1 = event.y - PEN_WIDTH // 2
        x2 = event.x + PEN_WIDTH // 2
        y2 = event.y + PEN_WIDTH // 2
        self.canvas.create_oval(x1, y1, x2, y2, fill=DRAW_COLOR, outline=DRAW_COLOR)
        self.draw_image.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw_image = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit between 0 and 9, then click Predict.")

    def predict(self):
        pixels = preprocess_image(self.image)
        prediction = self.model.predict(pixels)[0]
        probabilities = self.model.predict_proba(pixels)[0]
        confidence = probabilities[prediction] * 100

        top3 = sorted(
            enumerate(probabilities), key=lambda item: item[1], reverse=True
        )[:3]
        top_text = "   ".join(
            f"{digit}: {prob * 100:.1f}%" for digit, prob in top3
        )

        self.result_label.config(
            text=f"Predicted digit: {prediction} (confidence {confidence:.1f}%)\nTop guesses: {top_text}"
        )

    def run(self):
        self.root.resizable(False, False)
        self.root.mainloop()


def main():
    app = DigitRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()
