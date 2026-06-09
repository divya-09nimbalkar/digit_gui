# Handwritten Digit Recognition GUI

## Overview
This project implements a handwritten digit recognition GUI using Python and scikit-learn.
Draw a digit with the mouse, then click Predict to let the model identify the digit from 0 to 9.

## Features
- Interactive Tkinter drawing canvas
- Real machine learning model trained on the sklearn digits dataset
- MLP classifier with improved handwritten image preprocessing
- Predict button and confidence display
- Top-3 prediction scores shown after each guess
- Clear button to reset the canvas

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python digit_gui.py
```

## Notes
- The first run trains and caches a local model file named `digit_gui_model.joblib`.
- The model uses the `sklearn.datasets.load_digits` dataset and a KNN classifier.

## Author
Divya Nimbalkar
