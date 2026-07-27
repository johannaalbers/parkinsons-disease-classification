# Parkinson's disease classification from voice recordings

Supervised classification of Parkinson's disease from sustained phonation, and a demonstration of how much of the apparent performance comes from data leakage rather than from the disease signal.

**Headline result:** a k-nearest-neighbours classifier reaches 0.84 accuracy and 0.93 AUC when recordings are split at random. Split so that no speaker appears in both training and test data, the same model falls to 0.78 accuracy and 0.80 AUC, and specificity drops to 0.34. Roughly half of the model's apparent skill is speaker recognition, not disease detection.

---

## The problem with this dataset

The UCI Parkinsons dataset is usually described as 195 observations. It is not. It is **195 recordings from 32 people**, six or seven recordings each, of whom 24 have Parkinson's and 8 do not.

A random train/test split therefore puts recordings of the same voice on both sides of the split. A nearest-neighbour classifier can then match a test recording to another recording of the same person and inherit that person's label. Published accuracies on this dataset above 0.90 are frequently produced this way.

The unit of analysis is the speaker, not the recording. That is what this repository tests.

---

## Results

Both protocols use the same model: k-nearest neighbours, k = 15, all 22 acoustic features, z-scored inside a scikit-learn pipeline so that scaling parameters are fitted on training data only. Figures are the mean over 100 repeated splits, with the standard deviation across splits in brackets.

| | Record-level split | Subject-level split |
|---|---|---|
| Accuracy | 0.840 (0.037) | **0.778 (0.120)** |
| Sensitivity | 0.968 (0.031) | 0.945 (0.067) |
| Specificity | 0.443 (0.135) | **0.339 (0.313)** |
| AUC | 0.928 (0.032) | **0.800 (0.153)** |

**Predicting Parkinson's for every recording gives 0.754 accuracy.** The honest model beats that trivial baseline by 2.4 percentage points.

### How to read this

- **Sensitivity survives, specificity does not.** The model finds people with Parkinson's because it labels almost everyone as having Parkinson's. Specificity of 0.34 means two thirds of healthy speakers are wrongly flagged.
- **The standard deviation on specificity is 0.313.** With only 8 control speakers in the whole dataset, a subject-level test set contains two or three of them. Specificity moves in steps of a third. It is not a stable estimate and should not be quoted as one.
- **AUC falls by 0.128** under the honest protocol. That gap is the size of the leakage.
- Accuracy above 0.90 on this dataset, without a subject-aware split, should be read as a leakage artefact.

---

## What is not here

- No hyperparameter tuning. k = 15 is carried over from the original exercise. Tuning it would require a validation split nested inside the subject-level split, which 32 speakers cannot really support.
- No claim that this model is clinically useful. It is not. With 8 control speakers, no protocol produces a trustworthy specificity estimate.
- No ROC curves, despite them being straightforward to add, because a curve implies a precision the sample size does not support.

---

## Repository contents

| Path | Contents |
|---|---|
| `parkinsons_evaluation.py` | Reproduces the table above from the raw data |
| `Python/Parkinsons_disease.ipynb` | Original exploratory notebook: feature renaming, per-subject recording counts, scatter and box plots |
| `Data/parkinsons.data` | Raw UCI dataset |
| `PD_results/` | Exploratory figures |
| `results_split_comparison.csv` | Output of the evaluation script |

Run with:

```bash
pip install pandas numpy scikit-learn
python parkinsons_evaluation.py
```

---

## My contribution

Independent work, completed as part of coursework in machine learning and revised afterwards.

The original submission reported a single random split and quoted metrics that had been assembled from three different model fits, including a specificity value carried over from an in-sample confusion matrix. The revision here corrects that, adds the subject-level protocol, moves scaling inside a pipeline, and repeats every split 100 times so the uncertainty is visible.

---

## Data source

Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M. (2007). Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection. *BioMedical Engineering OnLine*, 6:23. Dataset available from the UCI Machine Learning Repository.
