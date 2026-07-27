"""
Parkinson's disease classification from sustained phonation: the effect of
subject-level data leakage on reported performance.

The UCI Parkinsons dataset holds 195 voice recordings from 32 people, six or
seven recordings each. A random train/test split therefore places recordings
from the same speaker on both sides of the split. The classifier can then
recognise the speaker rather than the disease.

This script quantifies that. It evaluates the same k-nearest-neighbours model
under two protocols:

  1. record-level split  (recordings shuffled at random, speakers shared)
  2. subject-level split (no speaker appears in both training and test data)

Run:  python parkinsons_evaluation.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score,
                             confusion_matrix)

DATA = "Data/parkinsons.data"
K = 15
N_REPEATS = 100
TEST_FRACTION = 0.25
RANDOM_STATE = 42


def load():
    df = pd.read_csv(DATA)
    # 'name' looks like phon_R01_S01_1: the S-block identifies the speaker.
    subject = df["name"].str.split("_", expand=True)[2]
    y = df["status"].astype(int)
    X = df.drop(columns=["name", "status"])
    return X, y, subject


def metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "auc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan,
        "n_test": len(y_true),
        "n_controls": tn + fp,
    }


def evaluate(X, y, groups, protocol, n_repeats=N_REPEATS):
    """Scaling sits inside the pipeline, so test data never informs the mean
    and standard deviation used to scale the training data."""
    rows = []
    for seed in range(n_repeats):
        if protocol == "subject":
            splitter = GroupShuffleSplit(1, test_size=TEST_FRACTION, random_state=seed)
            train, test = next(splitter.split(X, y, groups=groups))
        else:
            splitter = StratifiedShuffleSplit(1, test_size=TEST_FRACTION, random_state=seed)
            train, test = next(splitter.split(X, y))

        model = Pipeline([("scale", StandardScaler()),
                          ("knn", KNeighborsClassifier(n_neighbors=K))])
        model.fit(X.iloc[train], y.iloc[train])
        pred = model.predict(X.iloc[test])
        score = model.predict_proba(X.iloc[test])[:, 1]
        rows.append(metrics(y.iloc[test], pred, score))
    return pd.DataFrame(rows)


def summarise(name, df):
    print(f"\n{name}")
    print(f"  test set        {df.n_test.mean():.0f} recordings, "
          f"{df.n_controls.mean():.1f} of them controls")
    for m in ["accuracy", "sensitivity", "specificity", "auc"]:
        print(f"  {m:<15} {df[m].mean():.3f}  (sd {df[m].std():.3f})")


def main():
    X, y, subject = load()
    print(f"{len(X)} recordings from {subject.nunique()} speakers "
          f"({subject.groupby(subject).size().min()} to "
          f"{subject.groupby(subject).size().max()} each)")
    counts = y.groupby(subject).first().value_counts()
    print(f"speakers: {counts.get(1, 0)} with Parkinson's, {counts.get(0, 0)} controls")
    print(f"predicting Parkinson's for every recording gives accuracy {y.mean():.3f}")

    record = evaluate(X, y, subject, "record")
    subj = evaluate(X, y, subject, "subject")
    summarise("Record-level split (speakers shared between train and test)", record)
    summarise("Subject-level split (no speaker in both sets)", subj)

    print("\nDifference, subject-level minus record-level")
    for m in ["accuracy", "sensitivity", "specificity", "auc"]:
        print(f"  {m:<15} {subj[m].mean() - record[m].mean():+.3f}")

    out = pd.DataFrame({"record_level": record.mean(), "subject_level": subj.mean()})
    out.to_csv("results_split_comparison.csv")
    print("\nwritten: results_split_comparison.csv")


if __name__ == "__main__":
    main()
