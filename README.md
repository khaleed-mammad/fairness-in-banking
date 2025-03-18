# Fairness Project

## Overview
This project focuses on addressing fairness in machine learning within banking systems. Specifically, we aim to measure how different fairness algorithms impact the performance of various machine learning models. The task involves analyzing a marketing dataset from a financial institution to predict whether a client will subscribe to a term deposit. This is a binary classification problem, where the target variable represents subscription status.

## Related Work
Fairness in machine learning, especially within banking systems, has received significant attention over time. Prior research has explored various bias mitigation techniques, including preprocessing methods to ensure equitable outcomes across demographic groups. 

- **BeFair Toolkit (Castelnovo et al., 2021)**: A framework for identifying and mitigating biases in banking sector models.
- **Survey by Le Quy et al. (2022)**: Highlights the importance of diverse datasets in fairness-aware machine learning.
- **Borna B. (2023)**: Analyzes the impact of Disparate Impact Remover on LightGBM models.

## Data Preprocessing

### Dataset
We used the **Bank Marketing dataset** from the AIF360 library. The **protected attribute** in our study was **age**:
- **Privileged Group**: Individuals aged 25 and above
- **Unprivileged Group**: Individuals younger than 25

### Data Cleaning & Preprocessing
- Removed **10,700 missing entries**, leaving **30,488 samples**.
- Eliminated **40 duplicate rows**, reducing the dataset to **30,448 samples**.
- The dataset initially contained **58 features**, but after correlation analysis, **8 highly correlated features** were removed, leaving **50 features**.

### Feature Selection
- The dataset was **one-hot encoded**, except for the feature `duration`, which had **1,441 unique values**.
- Correlation analysis revealed several redundant features, which were **removed to prevent multicollinearity**.

## Modeling

### Chosen Algorithms
1. **Logistic Regression (LR)**: A simple, interpretable model used as a baseline.
2. **Random Forest (RF)**: Robust to overfitting and effective for structured data.
3. **Hist Gradient Boosting Classifier (HGB)**: A high-performing boosting model.

### Fairness Mitigation Techniques
We applied two **pre-processing** fairness mitigation techniques:
1. **Disparate Impact Remover (DIR)**: Adjusts feature values at different intensity levels to mitigate bias.
2. **Reweighing**: Assigns different sample weights based on demographic groups to balance outcomes.

### Hyperparameter Tuning
- **Random Forest** was fine-tuned using **Grid Search**, selecting:
{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}
- **Logistic Regression**: Set `max_iter=2000` to avoid non-convergence.
- **Hist Gradient Boosting**: Used default parameters.

### Evaluation Metrics
#### Performance Metrics:
- **Accuracy**
- **F1 Score**

#### Fairness Metrics:
- **Disparate Impact (DI)**: Ratio of favorable outcomes between groups (ideal value = 1).
- **Statistical Parity Difference (SPD)**: Difference in favorable outcomes between groups (ideal value = 0).

## Results

### Initial Dataset Fairness Evaluation
Before applying fairness techniques, the dataset showed **bias**:
- **Disparate Impact**: **1.86** (ideal value: **1**)
- **Statistical Parity Difference**: **0.11** (ideal value: **0**)

### Model Performance on Original Data
| Model | Accuracy (%) | F1 Score (%) | Disparate Impact | SPD |
|--------|-------------|-------------|------------------|-----|
| **Logistic Regression** | 89.75 | 50.42 | 1.857 | 0.064 |
| **Random Forest** | 90.22 | 54.97 | 1.488 | 0.042 |
| **Hist Gradient Boosting** | 90.60 | 60.43 | 1.451 | 0.048 |

### Model Performance After Applying **Disparate Impact Remover (DIR)**
| Model | Accuracy (%) | F1 Score (%) | Disparate Impact | SPD |
|--------|-------------|-------------|------------------|-----|
| **Logistic Regression** | 89.79 | 50.77 | 1.837 | 0.064 |
| **Random Forest** | 90.37 | 56.31 | 1.606 | 0.054 |
| **Hist Gradient Boosting** | 90.51 | 59.84 | 1.582 | 0.061 |

### Model Performance After Applying **Reweighing**
| Model | Accuracy (%) | F1 Score (%) | Disparate Impact | SPD |
|--------|-------------|-------------|------------------|-----|
| **Logistic Regression** | 89.84 | 50.95 | 1.174 | 0.0134 |
| **Random Forest** | 90.27 | 55.44 | 1.197 | 0.017 |
| **Hist Gradient Boosting** | 90.54 | 59.74 | 0.967 | -0.004 |

## Key Takeaways

1. **Hist Gradient Boosting Classifier** showed **the best performance** in terms of both **accuracy and fairness**.
2. **Disparate Impact Remover (DIR)** **negatively impacted fairness** in all cases.
3. **Reweighing** proved to be the **most effective fairness mitigation technique**, significantly reducing bias **without a major impact on model performance**.
4. **Fairness mitigation techniques do not necessarily degrade model performance**, and in some cases, they can slightly **improve performance** (e.g., Reweighing with Random Forest).

### Experiment Tracking with MLflow
- **MLflow** was used to log and track experiments, making it easier to compare different models and fairness techniques.

## Conclusion
- Preprocessing fairness techniques **can improve fairness** in machine learning models **without sacrificing accuracy**.
- **Reweighing** was the **best fairness intervention** in this study, significantly reducing bias.
- **Hist Gradient Boosting Classifier** outperformed other models in terms of **both accuracy and fairness**.

## References
- **Castelnovo, A., et al. (2021)** - BeFair: Addressing Fairness in the Banking Sector [[Link](https://doi.org/10.1109/BigData50022.2020.9377894)]
- **Le Quy, T., et al. (2022)** - A survey on datasets for fairness-aware machine learning [[Link](https://doi.org/10.1002/widm.1452)]
- **Dataset Source**: [AIF360 Bank Dataset](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.datasets.BankDataset.html)
