# CodeAlpha Credit Scoring Model

## Project Overview

This project builds a machine learning model to predict credit risk using the German Credit dataset.

The project compares three classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest

## Project Workflow

1. Load the credit dataset
2. Preprocess the data
3. Split the data into training and testing sets
4. Train three machine learning models
5. Evaluate model performance
6. Compare models using multiple evaluation metrics
7. Generate confusion matrices and ROC curves
8. Analyze feature importance
9. Save the best-performing model

## Model Performance

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The Random Forest model achieved the highest ROC-AUC score and was selected as the best model.

## Results

The project generates:

- Model comparison results
- Confusion matrices
- ROC curves
- Feature importance
- A saved trained model

All generated files are stored in the `results/` directory.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
