Predictive Customer Churn Analytics & Retention Decision Support System

A Streamlit-based final-year AI/ML project for customer churn analytics, model comparison, churn-risk prediction, explainability, and retention decision support.

Features

- CSV upload with dataset overview and preview
- Intelligent target/churn column detection
- Target validation to prevent continuous fields such as tenure from being used accidentally
- Identifier-column detection and exclusion review
- Missing-value handling through scikit-learn pipelines
- Leakage-safe train/validation/test workflow
- Model comparison using:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- XGBoost is not required for the first implementation
- Validation-based threshold optimization
- Final test-set metrics including accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and specificity
- Confusion matrix, ROC curve, precision-recall curve, churn distribution, and probability distribution
- Customer risk scoring with downloadable CSV reports
- Individual customer churn prediction with dynamic inputs from the uploaded dataset
- Explainability using model feature importance, logistic coefficients, or permutation importance fallback
- Downloadable model comparison, feature importance, customer risk, and model report files

How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Recommended Test Flow

1. Upload `churn_data-telco.csv`.
2. Confirm that `Churn` is detected as the target.
3. Try selecting `tenure` as the target and confirm that the app blocks it as suspicious.
4. Review detected identifier columns such as `customerID`.
5. Train the four built-in models.
6. Review validation comparison, final test metrics, confusion matrix, ROC curve, and PR curve.
7. Open Customer Risk and download the risk report.
8. Open Individual Prediction and confirm categorical fields use dropdowns.

Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
