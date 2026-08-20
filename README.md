# Pump It Up — Water Pump Classification

### Multiclass classification with data-quality analysis, feature engineering and CatBoost

This project predicts the operational status of water pumps using mixed numerical and categorical data.

It was developed as an **applied Machine Learning classification project based on a public competition dataset** and emphasizes **data cleaning, domain-aware treatment of missing values and masked zeros, feature engineering, model comparison and validation**.

---

## Project Highlights

- Multiclass classification: `functional`, `functional needs repair`, `non functional`
- Domain-aware treatment of missing values and masked zeros
- Feature engineering from temporal, geographical and infrastructure variables
- Comparison of Random Forest, HistGradientBoosting and CatBoost
- **CatBoost hold-out validation accuracy: 0.8107**
- **3-fold stratified CV mean accuracy: 0.8042 ± 0.0017**
- **Best public competition score: 0.8207**
- Explicit analysis of minority-class performance and model limitations

---

## Problem

The objective of the project is to predict the operational condition of water pumps from a dataset containing numerical, categorical, geographical and infrastructure-related information.

Each observation belongs to one of three classes:

- `functional`
- `functional needs repair`
- `non functional`

This is an imbalanced multiclass classification problem in which the minority `functional needs repair` class is substantially more difficult to identify than the two majority classes.

The project therefore focuses not only on overall accuracy, but also on data quality, class-level performance and validation stability.

---

## Data Quality and Preprocessing

A significant part of the workflow focused on understanding the quality and meaning of the input data before training the models.

The preprocessing process included:

- analysis of missing values
- identification of potentially masked missing values
- treatment of suspicious zero values
- categorical-value normalization
- high-cardinality categorical handling
- removal of redundant variables
- temporal feature extraction
- preservation of missingness information through indicator variables

Several numerical variables contained zero values that were unlikely to represent real observations.

These included variables such as:

- `gps_height`
- `longitude`
- `population`
- `construction_year`

Instead of treating every zero as a valid measurement, the workflow considered these values as potentially masked missing information and created additional indicators to preserve the fact that the original value had been zero.

---

## Feature Engineering

Additional variables were created to improve the information available to the models.

Examples include:

- year extracted from `date_recorded`
- month extracted from `date_recorded`
- estimated `pump_age`
- combined geographical variables
- missing-value indicators
- reduced representations of high-cardinality categorical variables

The estimated pump age was derived using the recorded year and construction year when both values were available and logically consistent.

These transformations were designed to convert raw competition variables into features with clearer analytical meaning.

---

## Machine Learning Approach

The project followed a structured modeling workflow:

1. Initial data-quality assessment
2. Missing-value analysis
3. Detection and treatment of masked zeros
4. Feature engineering
5. High-cardinality categorical handling
6. Stratified train/validation split
7. Model comparison
8. CatBoost optimization
9. Stratified cross-validation
10. Final model training
11. Competition submission and evaluation

Three main model families were evaluated:

- Random Forest
- HistGradientBoosting
- CatBoost

---

## Model Comparison

| Model | Validation accuracy |
|---|---:|
| CatBoost | **0.8107** |
| Random Forest | 0.7587 |
| HistGradientBoosting | 0.7387 |

CatBoost produced the strongest hold-out validation result.

It was particularly suitable for this problem because the dataset contains a large number of categorical variables together with numerical and engineered features.

---

## Cross-Validation

The selected CatBoost approach was also evaluated using stratified cross-validation.

The resulting performance was:

| Metric | Result |
|---|---:|
| Mean CV accuracy | **0.8042** |
| CV standard deviation | **0.0017** |

The relatively small standard deviation indicates that the model achieved similar performance across the evaluated folds.

This provided additional evidence that the hold-out validation result was not driven by a single favorable split.

---

## Validation Detail

For the selected CatBoost model:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| functional | 0.80 | 0.91 | 0.85 |
| functional needs repair | 0.59 | 0.28 | 0.38 |
| non functional | 0.86 | 0.77 | 0.81 |

The model performed particularly well when identifying `functional` and `non functional` pumps.

However, the minority `functional needs repair` class remained substantially more difficult.

Its recall of **0.28** means that many pumps requiring repair were classified as one of the other two states.

This is an important limitation and demonstrates why overall accuracy should not be considered in isolation for an imbalanced multiclass problem.

![CatBoost confusion matrix](images/confusion_matrix_catboost.png)

---

## Feature Importance

The final CatBoost model was also inspected using feature importance.

![CatBoost feature importance](images/feature_importance_catboost.png)

The strongest predictive signals included a combination of:

- geographical information
- water quantity information
- extraction characteristics
- waterpoint characteristics
- infrastructure information
- population
- temporal features
- engineered variables such as pump age

Feature importance describes how strongly variables contribute to the fitted model, but it should not be interpreted as evidence of causal relationships.

---

## Competition Result

The project was developed iteratively through multiple competition submissions.

The strongest public leaderboard result achieved was:

**0.8207**

The final documented notebook version produced:

**0.8183**

These two values are intentionally reported separately.

The best public score corresponds to the strongest individual competition submission, while the final notebook version was retained because it contained the more complete and structured workflow for:

- data-quality analysis
- feature engineering
- model comparison
- validation
- cross-validation
- model interpretation

![Competition score](images/competition_score.png)

---

## Main Lessons

Several important lessons emerged from the project:

- Data cleaning materially affected the modeling strategy.
- Masked zeros required domain-aware treatment rather than blanket imputation.
- CatBoost was particularly suitable for mixed numerical and categorical inputs.
- Feature engineering helped transform raw variables into more meaningful predictors.
- Stratification helped preserve class proportions during evaluation.
- Cross-validation provided additional evidence of model stability.
- Overall accuracy did not fully represent performance on the minority class.
- The `functional needs repair` class remained the main classification challenge.
- Competition score was treated as one signal rather than a substitute for validation quality.

---

## Repository Structure

```text
pump-it-up-water-pump-classification/
├── data/
│   └── README.md
├── images/
│   ├── competition_score.png
│   ├── confusion_matrix_catboost.png
│   └── feature_importance_catboost.png
├── notebooks/
│   └── pump_it_up_classification.ipynb
├── results/
│   ├── README.md
│   ├── classification_report.csv
│   ├── key_metrics.csv
│   └── model_comparison.csv
├── src/
│   ├── __init__.py
│   ├── modeling.py
│   └── preprocessing.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Public Portfolio Boundary

This repository intentionally excludes:

- original competition datasets
- row-level test predictions
- submission CSV files
- local file-system paths
- Google Colab-specific paths
- temporary development files

The public code is a cleaned portfolio refactor of the original academic workflow.

Reported metrics and competition results come from the completed original project.

---

## Data Policy

The original competition datasets are **not redistributed** in this public repository.

To reproduce the workflow, obtain the Pump It Up competition data from the original competition platform and place the required files inside the `data/` directory.

The expected data files are:

```text
train_values.csv
train_labels.csv
test_values.csv
```

These files are excluded from version control.

Row-level predictions and competition submission files are also not published.

---

## Reproducibility

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The repository includes reusable preprocessing and modeling components under:

```text
src/
```

The portfolio notebook is located at:

```text
notebooks/pump_it_up_classification.ipynb
```

Competition data must be obtained separately before reproducing the complete modeling workflow.

---

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- Random Forest
- HistGradientBoosting
- Matplotlib
- seaborn
- Jupyter
- Google Colab
- Git
- GitHub

---

## Limitations

- The target distribution is imbalanced.
- The `functional needs repair` class is substantially underrepresented.
- Minority-class recall remains significantly lower than performance on the majority classes.
- Overall accuracy therefore provides only a partial view of model quality.
- Competition leaderboard score is reported separately from local validation.
- Feature importance describes model behavior and should not be interpreted causally.
- The public repository does not redistribute the original competition data.

---

## Author

**Anastasia García Reziapova**

Machine Learning Project  
Data Science, Big Data & Business Analytics
