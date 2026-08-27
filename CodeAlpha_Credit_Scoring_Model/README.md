# CodeAlpha Task 1 — Credit Scoring Model

## Project overview

This project predicts an individual's credit risk from historical financial information. The target is binary: the model learns to distinguish customers classified as good credit risks from customers classified as bad credit risks. The project follows a complete machine-learning workflow: obtaining the dataset, inspecting the data, separating features from the target, preprocessing numerical and categorical variables, splitting the data into training and testing sets, training multiple classification algorithms, evaluating them with several metrics, comparing their performance, and saving the best model.

## Dataset

The project uses the **UCI Statlog (German Credit Data)** dataset. It contains 1,000 credit records and a mixture of categorical and numerical variables such as checking-account status, credit history, loan duration, credit amount, savings information, employment information, housing, age, existing credits, and other financial characteristics. The target supplied by the dataset has two classes: good credit and bad credit. The script downloads the dataset from the UCI repository automatically, so no manual CSV preparation is required.

## Data preprocessing

The dataset contains both numerical and categorical columns, so they cannot all be passed directly to a scikit-learn model. Numerical values are median-imputed if necessary and standardized with `StandardScaler`. Categorical values are filled with the most frequent category when necessary and converted into machine-readable binary columns with `OneHotEncoder`. A `ColumnTransformer` performs the two preprocessing paths while a `Pipeline` keeps preprocessing and model training together, which also helps prevent accidental leakage between the training and testing data.

## Train/test split

The data is divided into 80% training data and 20% testing data. The training portion is used to learn patterns, while the testing portion is kept separate until evaluation. `stratify=y` preserves approximately the same good/bad class proportion in both portions, which is useful because the classes are not perfectly balanced.

## Models

Three classification algorithms are compared. Logistic Regression provides a simple and interpretable baseline and is useful for understanding the relationship between features and the probability of a credit class. Decision Tree learns a series of if/then-style splits and can capture nonlinear relationships. Random Forest combines many decision trees and generally provides a stronger baseline for mixed tabular data because the ensemble reduces the instability of an individual tree.

## Evaluation metrics

Accuracy tells us the proportion of all predictions that are correct, but it should not be used alone for credit-risk classification. Precision tells us how often predicted positive/good-credit cases are actually good-credit cases. Recall measures how many of the actual positive/good-credit cases the model successfully finds. F1-score is the harmonic mean of precision and recall, so it provides a single balance between them. ROC-AUC measures how well the model ranks positive cases above negative cases across different classification thresholds; values closer to 1 indicate stronger discrimination.

## Confusion matrix and ROC curve

A confusion matrix shows true positives, true negatives, false positives, and false negatives. This is useful because different mistakes have different consequences in financial decision-making. The ROC curve plots the true-positive rate against the false-positive rate at different thresholds, while ROC-AUC summarizes the curve into one number. The script saves both plots for every model.

## Feature importance

After selecting the best model according to ROC-AUC, the script performs permutation importance. A feature is considered important when randomly shuffling that feature causes model performance to decrease. This gives a model-agnostic indication of which original variables are most useful to the prediction.

## Output

Running the script creates a `results/` directory containing a model-comparison CSV, confusion-matrix images, ROC-curve images, a top-feature-importance chart, a feature-importance CSV, and the serialized best model. These files can be shown in the GitHub repository as evidence of the completed work.

## How to run

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python credit_scoring.py
```

The machine running the script needs internet access the first time because the dataset is downloaded from UCI.

## Important project interpretation

This is an educational machine-learning project, not a real-world credit approval system. A production financial model would require extensive validation, fairness testing, regulatory review, explainability, security controls, monitoring, and careful handling of sensitive information. The project should therefore be presented as a demonstration of classification and model-evaluation techniques rather than as a system for making actual lending decisions.

