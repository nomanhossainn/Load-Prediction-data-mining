# Loan Approval Prediction with Fairness Analysis

### A Comparative Study of Preprocessing, Ensemble Learning, and Bias Mitigation

This project investigates machine learning techniques for predicting loan approval decisions while examining the impact of data preprocessing, ensemble learning, and demographic fairness. Using the **Analytics Vidhya Loan Prediction dataset**, we compare multiple classification models and evaluate whether removing sensitive demographic attributes can reduce gender-based disparities in loan approval outcomes.

The project was completed as part of the **Data Mining course, Spring 2026**, at United International University.

## 📌 Project Overview

Automated loan approval systems can improve the speed and consistency of financial decision-making. However, models trained on historical data may inherit existing biases and produce unfair outcomes for different demographic groups.

This study addresses three primary research questions:

1. How does full data preprocessing affect model performance compared with minimal preprocessing?
2. How do ensemble learning methods compare with individual classifiers?
3. Does removing sensitive attributes, such as gender and marital status, improve fairness, and what impact does it have on predictive performance?

The objective is to build and evaluate loan approval prediction models while considering both predictive performance and demographic fairness.

## ✨ Key Features

- **Exploratory Data Analysis (EDA):** Examines feature distributions, missing values, outliers, and class imbalance.
- **Comparative Preprocessing:** Evaluates minimal and full preprocessing pipelines.
- **Feature Engineering:** Creates financial features such as Total Income, EMI, Balance Income, and Debt-to-Income ratio.
- **Multiple Classification Models:** Compares four individual classifiers and three ensemble methods.
- **Class Imbalance Handling:** Applies SMOTE to the training data.
- **Fairness Analysis:** Compares gender-based approval rates and classification error rates.
- **Performance Evaluation:** Uses accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation results.
- **Responsible AI Considerations:** Discusses indirect discrimination, explainability, and algorithmic accountability.

## 📊 Dataset

The project uses the **Analytics Vidhya Loan Prediction Problem dataset**, available through Kaggle.

**[Find the Loan Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)**

The dataset represents a fictional housing finance company seeking to automate loan eligibility decisions.

| Dataset | Records | Purpose |
|---|---:|---|
| Training set | 614 | Model training and validation |
| Test set | 367 | Final inference |
| Features | 12 | Applicant financial and demographic information |
| Target | Loan Status | Approved (Y) or Not Approved (N) |

The training data contains missing values, skewed income distributions, outliers, and class imbalance. Approximately 69% of applications are approved and 31% are not approved.

### Main Features

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Loan Status (target)

## 🛠️ Technologies Used

- **Python**
- **Pandas** — Data manipulation
- **NumPy** — Numerical computation
- **Matplotlib / Seaborn** — Data visualization
- **Scikit-learn** — Machine learning, preprocessing, and evaluation
- **Imbalanced-learn** — SMOTE for class imbalance handling
- **Jupyter Notebook** — Experimentation and analysis

## 🔬 Methodology

The experimental workflow consists of five major stages.

### 1. Exploratory Data Analysis

- Analyze numerical and categorical features.
- Identify missing values and outliers.
- Examine the distribution of the target variable.
- Investigate relationships between applicant characteristics and loan approval.

### 2. Pipeline A — Minimal Preprocessing

The baseline pipeline includes:

- Mode imputation for categorical features.
- Median imputation for numerical features.
- Conversion of `Dependents` values from `3+` to `3`.
- Label encoding for categorical variables.

No feature scaling, outlier treatment, or class balancing is applied.

### 3. Pipeline B — Full Preprocessing

The comprehensive pipeline includes:

**Feature engineering**
- `TotalIncome` = ApplicantIncome + CoapplicantIncome
- `EMI` = LoanAmount / LoanAmountTerm
- `BalanceIncome` = TotalIncome − (EMI × 1000)
- `DebtToIncome` = LoanAmount / (TotalIncome + 1)

Log-transformed versions of selected skewed variables are also created.

**Data cleaning and transformation**
- Outlier capping using the 1.5 × IQR rule.
- KNN imputation with \(k = 5\) for numerical variables.
- One-hot encoding for nominal features.
- Ordinal encoding for education.
- StandardScaler for numerical features.
- SMOTE applied only to the training data.

The two pipelines are compared to investigate the contribution of preprocessing to model performance.

### 4. Model Training

Seven machine learning approaches are evaluated.

| Model | Category |
|---|---|
| Logistic Regression | Individual classifier |
| Random Forest | Individual classifier |
| Support Vector Machine (SVM) | Individual classifier |
| Multilayer Perceptron (MLP) | Neural network |
| Soft Voting | Ensemble learning |
| Stacking | Ensemble learning |
| Gradient Boosting | Ensemble learning |

The MLP uses three hidden layers containing 128, 64, and 32 neurons, respectively, with ReLU activation in the hidden layers.

### 5. Fairness Evaluation

A fairness-aware experiment removes `Gender` and `Married` from the input features and retrains the models.

The experiment compares:

- Male and female approval rates.
- Demographic approval-rate disparity.
- True Positive Rate (TPR).
- False Positive Rate (FPR).
- Predictive performance before and after removing sensitive attributes.

The objective is to assess whether removing direct demographic attributes reduces bias and whether indirect discrimination remains through correlated features.

## 📈 Experimental Results

### Individual Model Comparison

The following table presents the reported comparison between minimal and full preprocessing.

| Model | Minimal Accuracy | Full Accuracy | Full F1-Score | Full ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.18% | 85.37% | 89.41% | 0.8771 |
| Random Forest | 83.74% | 85.37% | 89.77% | 0.8554 |
| SVM | 69.11% | 84.55% | 89.14% | 0.8601 |
| MLP Neural Network | 69.11% | 77.24% | 83.33% | 0.8545 |

*Note: The table reproduces the main performance comparison in the report. Metrics may differ between experimental tables because the report includes multiple evaluation summaries.*

### Ensemble Learning Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Soft Voting | 85.37% | 86.02% | 94.12% | 89.89% | 0.8793 |
| Stacking | **86.99%** | 86.32% | **96.47%** | **91.11%** | 0.8724 |
| Gradient Boosting | 81.30% | 85.23% | 88.24% | 86.71% | 0.7867 |

**Best ensemble model: Stacking**

The Stacking ensemble achieved the highest reported accuracy, recall, and F1-score among the ensemble models in the main results table.

### Fairness Analysis

The Random Forest model was used as the primary model for the fairness comparison.

| Fairness Metric | Full Model | Fair Model |
|---|---:|---:|
| Male Approval Rate | 79.2% | 78.1% |
| Female Approval Rate | 48.0% | 56.0% |
| Approval-Rate Disparity | 31.2 percentage points | 22.1 percentage points |
| Male TPR | 94.3% | 95.7% |
| Female TPR | 85.7% | 92.9% |
| Male FPR | 38.5% | 30.8% |
| Female FPR | 0.0% | 9.1% |

Removing gender and marital status reduced the observed approval-rate disparity by **9.1 percentage points**. The experiment also reported improved accuracy and F1-score for the fairness-aware Random Forest model.

However, a disparity remained after sensitive attributes were removed, indicating that correlated features can still act as demographic proxies.

## 🔍 Key Findings

- Comprehensive preprocessing substantially improved SVM and MLP performance compared with the minimal pipeline.
- Feature engineering, scaling, imputation, and class imbalance handling were important components of the experimental workflow.
- Random Forest was a strong individual classifier, with a reported ROC-AUC of 0.866 in the comparative analysis.
- Stacking achieved a reported accuracy of approximately 87% and an F1-score of approximately 91%.
- Soft Voting achieved a strong ROC-AUC of 0.8793 in the main ensemble results.
- Removing sensitive demographic attributes reduced gender-based approval-rate disparity but did not eliminate it.
- Fairness cannot be guaranteed by removing sensitive features alone; correlated variables and historical bias require additional investigation.

## ⚠️ Limitations

The study identifies several limitations:

- The training dataset contains only 614 records.
- The dataset represents a fictional Indian housing finance company and may not generalize to other lending environments.
- The fairness comparison is limited to the binary gender categories available in the dataset.
- Systematic hyperparameter tuning was not performed.
- Random train-test splitting does not measure performance on future applications as effectively as temporal validation.
- Removing sensitive attributes does not eliminate proxy discrimination.
- The study does not establish that the models are ready for deployment in real financial decision-making systems.

## 🚀 Future Work

Potential directions for extending this project include:

- Applying adversarial debiasing and fairness-constrained optimization.
- Investigating reweighting methods to reduce demographic disparities.
- Performing temporal validation on future loan applications.
- Integrating SHAP or LIME for individual-level model explanations.
- Expanding fairness analysis to intersectional demographic groups.
- Conducting systematic hyperparameter optimization.
- Evaluating models on larger and more representative datasets.
- Monitoring demographic fairness throughout deployment.

## 👥 Project Team

**Data Mining — Spring 2026**

United International University (UIU)

- Md. Noman Hossain
- Foyez Ahammed Naeem
- Md. Annan
- Jubair Ahmed
- Md. Rafsan Bin Alam

## 📚 References

1. Analytics Vidhya. *Loan Prediction Problem Dataset*. [Kaggle Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.
3. Barocas, S., & Hardt, M. (2023). *Fairness and Machine Learning: Limitations and Opportunities*. MIT Press.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

## 📄 License

No license has been specified yet. Add a suitable open-source license if you intend to allow others to reuse, modify, or distribute the project.

---

**Disclaimer:** This project is an academic data mining study. Its results are experimental and should not be interpreted as financial advice or as evidence that the models are suitable for making real-world loan approval decisions without further validation, explainability, and fairness auditing.
