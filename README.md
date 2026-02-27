# parkinsons-disease-classification
Supervised classification of Parkinson’s disease using sklearn (feature preprocessing, model evaluation, ROC analysis).

## Project Overview

This project develops a supervised machine learning model to classify Parkinson’s disease based on biomedical voice measurements.

The objective is to distinguish between healthy individuals and patients with Parkinson’s disease using structured clinical features.

---

## Dataset

The dataset parkinsons.data contains biomedical voice measurements extracted from sustained phonations.

Target variable:
- Parkinson’s disease status (binary classification)

Features:
- Multiple acoustic and biomedical signal-derived measures

---

## Methodology

1. Data preprocessing
   - Feature cleaning
   - Train/test split
   - Feature scaling

2. Model training
   - Supervised classification models implemented using `scikit-learn`

3. Model evaluation
   - Precision
   - Sensitivity
   - Specificity
   - F-score
   - Accuracy

---

## Results

The trained models demonstrate strong discriminative performance in identifying Parkinson’s disease cases.

Key evaluation metrics:
Precision: 0.9142857142857143
Sensitivity (Recall): 1.0
Specificity: 0.6666666666666666
F-score: 0.9552238805970149
Accuracy:  0.8974358974358975

## My Contribution

This project was independently implemented as part of academic training in machine learning.

Responsibilities included:

- Data preprocessing and feature engineering
- Model implementation using `scikit-learn`
- Performance evaluation and interpretation
- Visualization of results

---

## Technologies Used

- Python
- NumPy
- Pandas
- scikit-learn
- Matplotlib / Seaborn

---


## 🎯 Skills Demonstrated

- Supervised learning
- Feature preprocessing and scaling
- Model evaluation (ROC, AUC, confusion matrix)
- Biomedical data analysis
- Reproducible machine learning workflow
