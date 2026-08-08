# Spam Email Classifier

A Python project for classifying email text as spam or ham using scikit-learn.

## Project structure

- `src/` - source code for training, prediction, and dataset loading
- `data/` - sample datasets and CSV files
- `models/` - saved model files
- `requirements.txt` - Python dependencies

## Setup

1. Create a Python virtual environment:

   ```bash
   python -m venv venv
   .\\venv\\Scripts\\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:

   ```bash
   python src/train.py
   ```

4. Predict using the saved model:

   ```bash
   python src/predict.py "Your email text here"
   ```

## Notes

- `train.py` uses a sample dataset at `data/spam.csv`.
- `model.joblib` is saved to `models/`.
- Modify the dataset or preprocessing logic as needed.
