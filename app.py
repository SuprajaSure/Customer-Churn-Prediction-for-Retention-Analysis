from __future__ import annotations

from itertools import product
from datetime import datetime
from io import StringIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


RANDOM_STATE = 42
CV_FOLDS = 5
BOOTSTRAP_SAMPLES = 500
CI_LEVEL = 0.95
HIGH_RISK_SHARE = 0.15
MEDIUM_RISK_SHARE = 0.25

LIKELY_TARGET_NAMES = {
    "churn",
    "exited",
    "attrition",
    "customerstatus",
    "status",
    "ischurn",
    "churned",
    "left",
    "exit",
    "retained",
}

POSITIVE_TARGET_TOKENS = {
    "yes",
    "y",
    "true",
    "1",
    "1.0",
    "churn",
    "churned",
    "exited",
    "left",
    "exit",
    "attrition",
}

NEGATIVE_TARGET_TOKENS = {
    "no",
    "n",
    "false",
    "0",
    "0.0",
    "stay",
    "stayed",
    "active",
    "retained",
    "notchurn",
    "nonchurn",
    "current",
}

IDENTIFIER_NAMES = {
    "id",
    "customerid",
    "customer_id",
    "customer id",
    "accountid",
    "account_id",
    "account id",
    "userid",
    "user_id",
    "user id",
    "clientid",
    "client_id",
    "client id",
    "subscriberid",
    "subscriber_id",
    "subscriber id",
    "rownumber",
    "row_number",
    "row number",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Predictive Customer Churn Analytics",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
<style>
    .stApp {
        background: #0b1220;
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: #020617;
        border-right: 1px solid #1e293b;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(135deg, #111827 0%, #31213a 55%, #831843 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 32px;
        margin-bottom: 18px;
    }
    .hero h1 {
        color: #fdf2f8;
        font-size: 2.25rem;
        line-height: 1.15;
        margin-bottom: 12px;
    }
    .hero p {
        color: #fbcfe8;
        font-size: 1.02rem;
        max-width: 920px;
    }
    .section-title {
        color: #f9a8d4;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1.2rem 0 0.6rem;
    }
    .metric-card {
        background: #020617;
        border: 1px solid #243244;
        border-left: 4px solid #ec4899;
        border-radius: 8px;
        padding: 16px;
        min-height: 108px;
    }
    .metric-card .label {
        color: #cbd5e1;
        font-size: 0.82rem;
        margin-bottom: 8px;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 700;
    }
    .subtle {
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    div[data-testid="stMetricValue"] {
        color: #fdf2f8;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def display_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="label">{label}</div>
    <div class="value">{value}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def canonical_target_value(value: Any) -> str:
    return str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def detect_target_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        if normalize_name(column) in LIKELY_TARGET_NAMES:
            return column
    return None


def assess_target_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    series = df[column]
    non_missing = series.dropna()
    unique_count = int(non_missing.nunique())
    unique_ratio = unique_count / max(len(non_missing), 1)
    normalized_name = normalize_name(column)
    warnings: list[str] = []
    blocking_errors: list[str] = []

    if len(non_missing) == 0:
        blocking_errors.append("The selected target column contains no usable values.")
    elif unique_count < 2:
        blocking_errors.append("The selected target column has only one class.")
    elif unique_count > 2:
        blocking_errors.append(
            "The selected target column has more than two classes. This first implementation supports binary churn classification."
        )

    if normalized_name not in LIKELY_TARGET_NAMES and unique_count == 2:
        warnings.append(
            "This column is binary, but its name is not a common churn target name. Confirm it is really the churn outcome."
        )

    if pd.api.types.is_numeric_dtype(series) and unique_count > 10:
        warnings.append(
            "This column appears to be numeric and continuous/high-cardinality, so it is probably not a churn target."
        )
        blocking_errors.append("Please choose a binary categorical churn target instead.")

    if unique_ratio > 0.25 and unique_count > 10:
        warnings.append(
            "This column has high cardinality, which is suspicious for a churn target."
        )
        blocking_errors.append("Please choose a target column with two meaningful classes.")

    if looks_like_identifier(df, column)[0]:
        warnings.append("This column looks like an identifier and should not be used as the target.")
        blocking_errors.append("Please choose an outcome column rather than an identifier.")

    return {
        "valid": len(blocking_errors) == 0,
        "warnings": list(dict.fromkeys(warnings)),
        "errors": list(dict.fromkeys(blocking_errors)),
        "unique_count": unique_count,
        "missing": int(series.isna().sum()),
    }


def infer_positive_class(series: pd.Series) -> Any:
    non_missing_values = list(series.dropna().unique())
    if not non_missing_values:
        return None

    for value in non_missing_values:
        token = canonical_target_value(value)
        if token in POSITIVE_TARGET_TOKENS:
            return value

    if len(non_missing_values) == 2:
        tokens = [canonical_target_value(value) for value in non_missing_values]
        if "1" in tokens:
            return non_missing_values[tokens.index("1")]

    return non_missing_values[-1]


def normalize_target(series: pd.Series, positive_class: Any) -> tuple[pd.Series, pd.Series]:
    valid_mask = series.notna()
    normalized = (
        series.loc[valid_mask]
        .astype(str)
        .str.strip()
        .eq(str(positive_class).strip())
        .astype(int)
    )
    return normalized, valid_mask


def looks_like_identifier(df: pd.DataFrame, column: str) -> tuple[bool, str]:
    series = df[column].dropna()
    if series.empty:
        return False, ""

    raw_name = str(column).strip().lower()
    normalized = normalize_name(column)
    unique_ratio = series.nunique() / max(len(series), 1)

    if raw_name in IDENTIFIER_NAMES or normalized in {normalize_name(name) for name in IDENTIFIER_NAMES}:
        return True, "identifier-like column name"

    if normalized.endswith("id") and unique_ratio > 0.80:
        return True, "name ends with ID and values are mostly unique"

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and len(numeric) > 2:
        sorted_values = np.sort(numeric.to_numpy())
        diffs = np.diff(sorted_values)
        if len(diffs) > 0 and np.all(diffs == 1):
            return True, "sequential numeric values"

    if series.dtype == object and unique_ratio > 0.95 and len(series) > 20:
        sample = series.astype(str).head(25)
        avg_length = sample.str.len().mean()
        has_digit = sample.str.contains(r"\d", regex=True).mean() > 0.5
        if avg_length >= 4 and has_digit:
            return True, "highly unique identifier-like text"

    return False, ""


def identify_id_columns(df: pd.DataFrame, target_col: str | None = None) -> dict[str, str]:
    detected: dict[str, str] = {}
    for column in df.columns:
        if column == target_col:
            continue
        is_id, reason = looks_like_identifier(df, column)
        if is_id:
            detected[column] = reason
    return detected


def convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for column in converted.columns:
        if pd.api.types.is_object_dtype(converted[column]) or pd.api.types.is_string_dtype(converted[column]):
            cleaned = converted[column].replace(r"^\s*$", np.nan, regex=True)
            numeric = pd.to_numeric(cleaned, errors="coerce")
            parse_ratio = numeric.notna().sum() / max(cleaned.notna().sum(), 1)
            if parse_ratio >= 0.90 and numeric.notna().sum() > 0:
                converted[column] = numeric
    return converted


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_features, categorical_features = split_feature_types(X)

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("num", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def build_model_specs(X_train: pd.DataFrame) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "Logistic Regression": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", build_preprocessor(X_train, scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            solver="liblinear",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "search_space": {
                "model__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
            "use_sample_weight": False,
            "imbalance_strategy": "class_weight='balanced'",
            "max_candidates": 5,
            "available": True,
        },
        "Decision Tree": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        DecisionTreeClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "search_space": {
                "model__max_depth": [4, 6, 8, 10, None],
                "model__min_samples_leaf": [5, 10, 20],
            },
            "use_sample_weight": False,
            "imbalance_strategy": "class_weight='balanced'",
            "max_candidates": 10,
            "available": True,
        },
        "Random Forest": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "search_space": {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [2, 4, 8],
            },
            "use_sample_weight": False,
            "imbalance_strategy": "class_weight='balanced'",
            "max_candidates": 10,
            "available": True,
        },
        "Gradient Boosting": {
            "pipeline": Pipeline(
                [
                    ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        GradientBoostingClassifier(random_state=RANDOM_STATE),
                    ),
                ]
            ),
            "search_space": {
                "model__n_estimators": [100, 150, 200],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__min_samples_leaf": [5, 10, 20],
                "model__subsample": [0.8, 1.0],
            },
            "use_sample_weight": True,
            "imbalance_strategy": "balanced sample weights passed during fit",
            "max_candidates": 12,
            "available": True,
        },
    }

    if XGBClassifier is not None:
        specs["XGBoost"] = {
            "pipeline": Pipeline(
                [
                    ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            tree_method="hist",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "search_space": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [3, 4, 5],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__gamma": [0, 0.1],
                "model__reg_lambda": [1.0, 2.0],
            },
            "use_sample_weight": True,
            "imbalance_strategy": "balanced sample weights passed during fit",
            "max_candidates": 12,
            "available": True,
        }
    else:
        specs["XGBoost"] = {
            "available": False,
            "failure": "xgboost is not installed in this environment.",
        }

    if LGBMClassifier is not None:
        specs["LightGBM"] = {
            "pipeline": Pipeline(
                [
                    ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        LGBMClassifier(
                            objective="binary",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
            "search_space": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
                "model__max_depth": [-1, 4, 6],
                "model__min_child_samples": [10, 20, 40],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__reg_lambda": [0.0, 1.0],
            },
            "use_sample_weight": True,
            "imbalance_strategy": "balanced sample weights passed during fit",
            "max_candidates": 12,
            "available": True,
        }
    else:
        specs["LightGBM"] = {
            "available": False,
            "failure": "lightgbm is not installed in this environment.",
        }

    if CatBoostClassifier is not None:
        specs["CatBoost"] = {
            "pipeline": Pipeline(
                [
                    ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        CatBoostClassifier(
                            loss_function="Logloss",
                            verbose=False,
                            allow_writing_files=False,
                            random_seed=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "search_space": {
                "model__iterations": [100, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__depth": [4, 6, 8],
                "model__l2_leaf_reg": [1, 3, 5, 7],
            },
            "use_sample_weight": True,
            "imbalance_strategy": "balanced sample weights passed during fit",
            "max_candidates": 10,
            "available": True,
        }
    else:
        specs["CatBoost"] = {
            "available": False,
            "failure": "catboost is not installed in this environment.",
        }

    return specs


def build_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        name: spec["pipeline"]
        for name, spec in build_model_specs(X_train).items()
        if spec.get("available")
    }


def parameter_candidates(
    search_space: dict[str, list[Any]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    if not search_space:
        return [{}]

    keys = list(search_space)
    candidates = [dict(zip(keys, values)) for values in product(*(search_space[key] for key in keys))]
    if len(candidates) <= max_candidates:
        return candidates

    rng = np.random.default_rng(RANDOM_STATE)
    selected = rng.choice(len(candidates), size=max_candidates, replace=False)
    return [candidates[int(index)] for index in selected]


def fit_with_optional_weights(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_sample_weight: bool,
) -> Pipeline:
    if use_sample_weight:
        counts = y_train.value_counts().to_dict()
        weights = y_train.map(lambda value: len(y_train) / (2 * counts[value])).to_numpy()
        pipeline.fit(X_train, y_train, model__sample_weight=weights)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate_candidate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    threshold_objective: str,
    use_sample_weight: bool,
) -> dict[str, Any]:
    fold_rows = []
    oof_prob = np.zeros(len(y), dtype=float)
    y_array = y.to_numpy()

    for fold_index, (train_index, val_index) in enumerate(cv.split(X, y), start=1):
        fold_pipeline = clone(pipeline)
        X_fold_train = X.iloc[train_index]
        y_fold_train = y.iloc[train_index]
        X_fold_val = X.iloc[val_index]
        y_fold_val = y.iloc[val_index]

        fit_with_optional_weights(fold_pipeline, X_fold_train, y_fold_train, use_sample_weight)
        val_prob = fold_pipeline.predict_proba(X_fold_val)[:, 1]
        oof_prob[val_index] = val_prob
        fold_threshold, _ = optimize_threshold(y_fold_val, val_prob, threshold_objective)
        fold_metrics = calculate_metrics(y_fold_val, val_prob, fold_threshold)
        fold_metrics["Fold"] = fold_index
        fold_rows.append(fold_metrics)

    fold_df = pd.DataFrame(fold_rows)
    summary: dict[str, Any] = {
        "folds": fold_df,
        "oof_prob": oof_prob,
        "y_oof": y_array,
    }
    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "ROC-AUC", "PR-AUC"]:
        summary[metric] = float(fold_df[metric].mean())
        summary[f"{metric} Std"] = float(fold_df[metric].std(ddof=1))
    summary["Threshold"] = float(fold_df["Threshold"].mean())
    return summary


def out_of_fold_probabilities(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    use_sample_weight: bool,
) -> np.ndarray:
    oof_prob = np.zeros(len(y), dtype=float)
    for train_index, val_index in cv.split(X, y):
        fold_pipeline = clone(pipeline)
        fit_with_optional_weights(
            fold_pipeline,
            X.iloc[train_index],
            y.iloc[train_index],
            use_sample_weight,
        )
        oof_prob[val_index] = fold_pipeline.predict_proba(X.iloc[val_index])[:, 1]
    return oof_prob


def calculate_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "ROC-AUC": np.nan,
        "PR-AUC": np.nan,
        "Threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    if len(np.unique(y_true)) == 2:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        metrics["PR-AUC"] = average_precision_score(y_true, y_prob)

    return metrics


def optimize_threshold(y_true: pd.Series, y_prob: np.ndarray, objective: str) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.10, 0.90, 81):
        metrics = calculate_metrics(y_true, y_prob, float(threshold))
        balanced = np.nanmean([metrics["Recall"], metrics["Specificity"]])
        rows.append(
            {
                "Threshold": float(threshold),
                "F1 Score": metrics["F1 Score"],
                "Recall": metrics["Recall"],
                "Precision": metrics["Precision"],
                "Specificity": metrics["Specificity"],
                "Balanced Score": balanced,
            }
        )

    threshold_df = pd.DataFrame(rows)
    if objective == "Recall":
        sort_columns = ["Recall", "F1 Score", "Precision"]
    elif objective == "Balanced performance":
        sort_columns = ["Balanced Score", "F1 Score", "Recall"]
    else:
        sort_columns = ["F1 Score", "Recall", "Precision"]

    best_row = threshold_df.sort_values(sort_columns, ascending=False).iloc[0]
    return float(best_row["Threshold"]), threshold_df


def apply_probability_calibration(
    y_train: pd.Series,
    oof_prob: np.ndarray,
    test_prob: np.ndarray,
    method: str,
) -> dict[str, Any]:
    before_brier = brier_score_loss(y_train, oof_prob)
    calibrated_oof = oof_prob.copy()
    calibrated_test = test_prob.copy()
    status = "Not applied"

    if method == "Platt scaling":
        calibrator = LogisticRegression(solver="liblinear", random_state=RANDOM_STATE)
        calibrator.fit(oof_prob.reshape(-1, 1), y_train)
        calibrated_oof = calibrator.predict_proba(oof_prob.reshape(-1, 1))[:, 1]
        calibrated_test = calibrator.predict_proba(test_prob.reshape(-1, 1))[:, 1]
        status = "Applied Platt scaling using training out-of-fold probabilities"
    elif method == "Isotonic regression":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof_prob, y_train)
        calibrated_oof = calibrator.predict(oof_prob)
        calibrated_test = calibrator.predict(test_prob)
        status = "Applied isotonic regression using training out-of-fold probabilities"

    after_brier = brier_score_loss(y_train, calibrated_oof)
    improved = after_brier <= before_brier

    return {
        "oof_prob": calibrated_oof if improved else oof_prob,
        "test_prob": calibrated_test if improved else test_prob,
        "table": pd.DataFrame(
            [
                {
                    "Calibration": "Before calibration",
                    "Brier Score": before_brier,
                    "Used for Final Probabilities": method == "None",
                },
                {
                    "Calibration": method,
                    "Brier Score": after_brier,
                    "Used for Final Probabilities": method != "None" and improved,
                },
            ]
        ),
        "status": status if improved or method == "None" else f"{status}; retained uncalibrated probabilities because Brier score did not improve",
        "improved": bool(improved),
    }


def bootstrap_confidence_intervals(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    n_samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = CI_LEVEL,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    y_values = y_true.to_numpy()
    metric_values: dict[str, list[float]] = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Specificity": [],
        "ROC-AUC": [],
        "PR-AUC": [],
    }

    for _ in range(n_samples):
        indices = rng.integers(0, len(y_values), len(y_values))
        sample_y = pd.Series(y_values[indices])
        sample_prob = y_prob[indices]
        if sample_y.nunique() < 2:
            continue
        metrics = calculate_metrics(sample_y, sample_prob, threshold)
        for metric in metric_values:
            if not pd.isna(metrics[metric]):
                metric_values[metric].append(float(metrics[metric]))

    alpha = 1 - confidence
    rows = []
    point_metrics = calculate_metrics(y_true, y_prob, threshold)
    for metric, values in metric_values.items():
        if values:
            rows.append(
                {
                    "Metric": metric,
                    "Estimate": point_metrics[metric],
                    "CI Low": float(np.quantile(values, alpha / 2)),
                    "CI High": float(np.quantile(values, 1 - alpha / 2)),
                    "Bootstrap Samples": len(values),
                }
            )
    return pd.DataFrame(rows)


def threshold_sensitivity_table(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    rows = []
    for threshold in thresholds:
        metrics = calculate_metrics(y_true, y_prob, float(threshold))
        rows.append(
            {
                "Threshold": float(threshold),
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
                "Specificity": metrics["Specificity"],
                "Targeted Customers": int(metrics["TP"] + metrics["FP"]),
                "False Positives": int(metrics["FP"]),
                "False Negatives": int(metrics["FN"]),
            }
        )
    return pd.DataFrame(rows)


def derive_risk_boundaries(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    low_boundary = float(np.quantile(y_prob, 1 - HIGH_RISK_SHARE - MEDIUM_RISK_SHARE))
    high_boundary = float(np.quantile(y_prob, 1 - HIGH_RISK_SHARE))
    if high_boundary <= low_boundary:
        high_boundary = min(0.95, low_boundary + 0.10)
    return {
        "low_medium": low_boundary,
        "medium_high": high_boundary,
    }


def assign_risk_level_with_boundaries(probability: float, boundaries: dict[str, float]) -> str:
    if probability >= boundaries["medium_high"]:
        return "High"
    if probability >= boundaries["low_medium"]:
        return "Medium"
    return "Low"


def risk_tier_analysis(
    y_true: pd.Series,
    y_prob: np.ndarray,
    boundaries: dict[str, float],
) -> pd.DataFrame:
    tiers = [assign_risk_level_with_boundaries(probability, boundaries) for probability in y_prob]
    analysis_df = pd.DataFrame(
        {
            "Actual": y_true.to_numpy(),
            "Churn Probability": y_prob,
            "Risk Level": tiers,
        }
    )
    total_churn = max(int(analysis_df["Actual"].sum()), 1)
    rows = []
    for tier in ["Low", "Medium", "High"]:
        tier_df = analysis_df[analysis_df["Risk Level"] == tier]
        count = len(tier_df)
        actual_churn = int(tier_df["Actual"].sum()) if count else 0
        rows.append(
            {
                "Risk Level": tier,
                "Customers": count,
                "Dataset Share": count / len(analysis_df) if len(analysis_df) else np.nan,
                "Mean Probability": tier_df["Churn Probability"].mean() if count else np.nan,
                "Actual Churn Rate": actual_churn / count if count else np.nan,
                "Recall Contribution": actual_churn / total_churn,
            }
        )
    return pd.DataFrame(rows)


def business_threshold_scenarios(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    scenarios = [
        {"Scenario": "Low intervention cost", "Retention Cost Units": 1.0, "Churn Loss Units": 5.0},
        {"Scenario": "Medium intervention cost", "Retention Cost Units": 1.0, "Churn Loss Units": 3.0},
        {"Scenario": "High intervention cost", "Retention Cost Units": 1.0, "Churn Loss Units": 2.0},
    ]
    rows = []
    for scenario in scenarios:
        best: dict[str, Any] | None = None
        for threshold in np.arange(0.10, 0.91, 0.01):
            metrics = calculate_metrics(y_true, y_prob, float(threshold))
            targeted = int(metrics["TP"] + metrics["FP"])
            net_value = metrics["TP"] * scenario["Churn Loss Units"] - targeted * scenario["Retention Cost Units"]
            candidate = {
                **scenario,
                "Optimal Threshold": float(threshold),
                "Expected Net Value Units": float(net_value),
                "Targeted Customers": targeted,
                "True Churners Targeted": int(metrics["TP"]),
                "False Positives": int(metrics["FP"]),
            }
            if best is None or candidate["Expected Net Value Units"] > best["Expected Net Value Units"]:
                best = candidate
        if best:
            rows.append(best)
    return pd.DataFrame(rows)


def hyperparameter_rows(
    model_name: str,
    search_space: dict[str, list[Any]],
    best_params: dict[str, Any],
    selection_method: str,
) -> list[dict[str, Any]]:
    rows = []
    for parameter, values in search_space.items():
        clean_parameter = parameter.replace("model__", "")
        rows.append(
            {
                "Model": model_name,
                "Hyperparameter": clean_parameter,
                "Search Range": ", ".join(str(value) for value in values),
                "Final Value": best_params.get(parameter, "default"),
                "Selection Method": selection_method,
            }
        )
    return rows


def build_modeling_data(
    df: pd.DataFrame,
    target_col: str,
    positive_class: Any,
    excluded_id_columns: list[str],
) -> dict[str, Any]:
    y, target_mask = normalize_target(df[target_col], positive_class)
    modeling_df = df.loc[target_mask].copy()
    X = modeling_df.drop(columns=[target_col] + excluded_id_columns, errors="ignore")
    X = convert_numeric_like_columns(X)

    constant_columns = [column for column in X.columns if X[column].nunique(dropna=True) <= 1]
    if constant_columns:
        X = X.drop(columns=constant_columns)

    return {
        "X": X,
        "y": y,
        "constant_columns": constant_columns,
        "missing_numeric": int(X.select_dtypes(include=["number", "bool"]).isna().sum().sum()),
        "missing_categorical": int(X.select_dtypes(exclude=["number", "bool"]).isna().sum().sum()),
    }


def safe_train_test_split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, ...]:
    class_counts = y.value_counts()
    if len(class_counts) != 2:
        raise ValueError("The target must contain exactly two classes.")
    if class_counts.min() < 5 or len(y) < 30:
        raise ValueError(
            "There are too few records or too few churn examples for a reliable train/test split."
        )

    stratify = y if class_counts.min() >= 5 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=stratify,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def safe_train_validation_test_split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, ...]:
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
    return X_train, pd.DataFrame(), X_test, y_train, pd.Series(dtype=int), y_test


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    manual_threshold: float,
    optimize: bool,
    threshold_objective: str,
    selection_priority: str,
    calibration_method: str = "Platt scaling",
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
    min_class_count = int(y_train.value_counts().min())
    n_splits = min(CV_FOLDS, min_class_count)
    if n_splits < 2:
        raise ValueError("There are too few records in one class for cross-validation.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    model_specs = build_model_specs(X_train)
    trained_models: dict[str, Pipeline] = {}
    validation_rows = []
    test_rows = []
    thresholds: dict[str, float] = {}
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}
    fold_tables: dict[str, pd.DataFrame] = {}
    best_params_by_model: dict[str, dict[str, Any]] = {}
    imbalance_strategies: dict[str, str] = {}
    calibration_tables: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    hyperparameter_table_rows: list[dict[str, Any]] = []

    for name, spec in model_specs.items():
        if not spec.get("available"):
            failures[name] = spec.get("failure", "Model library is not available.")
            continue

        try:
            base_pipeline: Pipeline = spec["pipeline"]
            search_space = spec.get("search_space", {})
            use_sample_weight = bool(spec.get("use_sample_weight", False))
            imbalance_strategies[name] = str(spec.get("imbalance_strategy", "None"))
            candidates = parameter_candidates(search_space, int(spec.get("max_candidates", 1)))

            candidate_rows = []
            candidate_summaries = []
            for params in candidates:
                candidate_pipeline = clone(base_pipeline)
                candidate_pipeline.set_params(**params)
                cv_summary = cross_validate_candidate(
                    candidate_pipeline,
                    X_train,
                    y_train,
                    cv,
                    threshold_objective,
                    use_sample_weight,
                )
                candidate_summaries.append((params, cv_summary))
                candidate_rows.append(
                    {
                        "Parameters": params,
                        **{
                            metric: cv_summary[metric]
                            for metric in [
                                "Accuracy",
                                "Precision",
                                "Recall",
                                "F1 Score",
                                "Specificity",
                                "ROC-AUC",
                                "PR-AUC",
                            ]
                        },
                    }
                )

            priority_map = {
                "F1 Score": ["F1 Score", "Recall", "ROC-AUC"],
                "Recall": ["Recall", "F1 Score", "ROC-AUC"],
                "ROC-AUC": ["ROC-AUC", "F1 Score", "Recall"],
            }
            sort_columns = priority_map.get(selection_priority, priority_map["F1 Score"])
            candidate_df = pd.DataFrame(candidate_rows)
            ranking = candidate_df.fillna(-1).sort_values(sort_columns, ascending=False)
            best_index = int(ranking.index[0])
            best_params, best_cv_summary = candidate_summaries[best_index]

            best_pipeline = clone(base_pipeline)
            best_pipeline.set_params(**best_params)
            oof_prob = out_of_fold_probabilities(best_pipeline, X_train, y_train, cv, use_sample_weight)
            chosen_threshold = manual_threshold

            if optimize:
                chosen_threshold, threshold_table = optimize_threshold(
                    y_train, oof_prob, threshold_objective
                )
                threshold_tables[name] = threshold_table

            final_pipeline = clone(best_pipeline)
            fit_with_optional_weights(final_pipeline, X_train, y_train, use_sample_weight)
            test_prob = final_pipeline.predict_proba(X_test)[:, 1]

            calibration_result = apply_probability_calibration(
                y_train,
                oof_prob,
                test_prob,
                calibration_method,
            )
            final_oof_prob = calibration_result["oof_prob"]
            final_test_prob = calibration_result["test_prob"]
            calibration_tables[name] = calibration_result["table"]

            if optimize and calibration_method != "None" and calibration_result["improved"]:
                chosen_threshold, threshold_table = optimize_threshold(
                    y_train, final_oof_prob, threshold_objective
                )
                threshold_tables[name] = threshold_table

            cv_metrics = calculate_metrics(y_train, final_oof_prob, chosen_threshold)
            test_metrics = calculate_metrics(y_test, final_test_prob, chosen_threshold)

            validation_rows.append(
                {
                    "Model": name,
                    **cv_metrics,
                    **{
                        f"{metric} Std": best_cv_summary[f"{metric} Std"]
                        for metric in [
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "F1 Score",
                            "Specificity",
                            "ROC-AUC",
                            "PR-AUC",
                        ]
                    },
                    "CV Folds": n_splits,
                    "Imbalance Handling": imbalance_strategies[name],
                    "Calibration": calibration_result["status"],
                }
            )
            test_rows.append({"Model": name, **test_metrics})
            thresholds[name] = chosen_threshold
            probabilities[name] = {"validation": final_oof_prob, "test": final_test_prob, "raw_oof": oof_prob}
            trained_models[name] = final_pipeline
            fold_tables[name] = best_cv_summary["folds"]
            best_params_by_model[name] = best_params
            hyperparameter_table_rows.extend(
                hyperparameter_rows(
                    name,
                    search_space,
                    best_params,
                    f"{n_splits}-fold stratified CV on the training data",
                )
            )
        except Exception as exc:
            failures[name] = str(exc)

    if not trained_models:
        raise RuntimeError("All model training attempts failed.")

    validation_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)

    priority_map = {
        "F1 Score": ["F1 Score", "Recall", "ROC-AUC"],
        "Recall": ["Recall", "F1 Score", "ROC-AUC"],
        "ROC-AUC": ["ROC-AUC", "F1 Score", "Recall"],
    }
    sort_columns = priority_map.get(selection_priority, priority_map["F1 Score"])
    ranking = validation_df.fillna(-1).sort_values(sort_columns, ascending=False)
    best_model_name = str(ranking.iloc[0]["Model"])
    best_threshold = float(thresholds[best_model_name])
    best_oof_prob = probabilities[best_model_name]["validation"]
    best_test_prob = probabilities[best_model_name]["test"]
    risk_boundaries = derive_risk_boundaries(y_train, best_oof_prob)
    confidence_rows = []
    for model_name, probs in probabilities.items():
        ci_df = bootstrap_confidence_intervals(y_test, probs["test"], thresholds[model_name])
        ci_df.insert(0, "Model", model_name)
        confidence_rows.append(ci_df)
    confidence_intervals = pd.concat(confidence_rows, ignore_index=True) if confidence_rows else pd.DataFrame()

    return {
        "models": trained_models,
        "validation_comparison": validation_df,
        "cv_comparison": validation_df,
        "test_comparison": test_df,
        "best_model_name": best_model_name,
        "thresholds": thresholds,
        "threshold_tables": threshold_tables,
        "fold_tables": fold_tables,
        "best_params": best_params_by_model,
        "imbalance_strategies": imbalance_strategies,
        "hyperparameter_table": pd.DataFrame(hyperparameter_table_rows),
        "calibration_tables": calibration_tables,
        "confidence_intervals": confidence_intervals,
        "threshold_sensitivity": threshold_sensitivity_table(y_train, best_oof_prob),
        "business_scenarios": business_threshold_scenarios(y_train, best_oof_prob),
        "risk_boundaries": risk_boundaries,
        "risk_tier_validation": risk_tier_analysis(y_test, best_test_prob, risk_boundaries),
        "probabilities": probabilities,
        "failures": failures,
        "splits": {
            "X_train": X_train,
            "X_val": pd.DataFrame(),
            "X_test": X_test,
            "y_train": y_train,
            "y_val": pd.Series(dtype=int),
            "y_test": y_test,
        },
        "cv_folds": n_splits,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "ci_level": CI_LEVEL,
    }


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    percent_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "Specificity",
    ]
    percent_columns += [column for column in display_df.columns if column.endswith(" Std")]
    for column in percent_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(pct)
    if "Threshold" in display_df.columns:
        display_df["Threshold"] = display_df["Threshold"].map(lambda value: f"{value:.2f}")
    return display_df


def assign_risk_level(probability: float, boundaries: dict[str, float] | None = None) -> str:
    if boundaries is not None:
        return assign_risk_level_with_boundaries(probability, boundaries)
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        names = preprocessor.get_feature_names_out()
        return [clean_feature_name(name) for name in names]
    except Exception:
        return []


def clean_feature_name(name: str) -> str:
    cleaned = name.replace("num__", "").replace("cat__", "")
    return cleaned.replace("_", " ")


def get_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int,
) -> tuple[pd.DataFrame, str]:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    if hasattr(model, "feature_importances_") and feature_names:
        values = model.feature_importances_
        method = "Model feature importance"
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": values})
    elif hasattr(model, "coef_") and feature_names:
        values = np.abs(model.coef_[0])
        method = "Absolute logistic regression coefficients"
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": values})
    else:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="f1",
        )
        method = "Permutation importance on the test split"
        importance_df = pd.DataFrame(
            {"Feature": X_test.columns, "Importance": result.importances_mean}
        )

    return (
        importance_df.sort_values("Importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True),
        method,
    )


def plot_churn_distribution(y: pd.Series) -> plt.Figure:
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    labels = ["Non-Churn", "Churn"]
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=140)
    sns.barplot(x=labels, y=counts.values, palette=["#38bdf8", "#ec4899"], ax=ax)
    total = counts.sum()
    for index, count in enumerate(counts.values):
        ax.text(index, count, f"{count:,}\n{count / total:.1%}", ha="center", va="bottom")
    ax.set_ylabel("Customers")
    ax.set_xlabel("")
    ax.set_title("Target Class Distribution")
    fig.tight_layout()
    return fig


def plot_probability_distribution(y_prob: np.ndarray, threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=140)
    sns.histplot(y_prob, bins=20, kde=True, color="#f472b6", ax=ax)
    ax.axvline(threshold, color="#facc15", linestyle="--", linewidth=2, label=f"Threshold {threshold:.2f}")
    ax.set_xlabel("Predicted churn probability")
    ax.set_title("Churn Probability Distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> plt.Figure:
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=140)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="RdPu",
        cbar=False,
        xticklabels=["Predicted Non-Churn", "Predicted Churn"],
        yticklabels=["Actual Non-Churn", "Actual Churn"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curves(results: dict[str, Any]) -> plt.Figure:
    y_test = results["splits"]["y_test"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    for name, probs in results["probabilities"].items():
        y_prob = probs["test"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pr_curve(y_true: pd.Series, y_prob: np.ndarray) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.plot(recall, precision, color="#ec4899", label=f"PR-AUC {pr_auc:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_calibration(y_true: pd.Series, y_prob: np.ndarray) -> plt.Figure:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=140)
    ax.plot(prob_pred, prob_true, marker="o", color="#ec4899", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    return fig


def outlier_summary(df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    numeric_df = convert_numeric_like_columns(df).select_dtypes(include=["number"])

    rows = []

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((series < lower) | (series > upper)).sum()
        )

        if count > 0:
            rows.append(
                {
                    "Feature": column,
                    "Potential Outliers": count,
                    "Share": count / len(series),
                }
            )

    # Important: preserve the expected columns even when
    # no outliers are detected.
    if not rows:
        return pd.DataFrame(
            columns=["Feature", "Potential Outliers", "Share"]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Potential Outliers", ascending=False)
        .head(max_rows)
        .reset_index(drop=True)
    )


def create_risk_table(df: pd.DataFrame, app_state: dict[str, Any], threshold: float) -> pd.DataFrame:
    results = app_state["training_results"]
    model = results["models"][results["best_model_name"]]
    boundaries = results.get("risk_boundaries")
    feature_columns = app_state["feature_columns"]
    X_all = convert_numeric_like_columns(
        df.drop(
            columns=[app_state["target_col"]] + app_state["excluded_id_columns"] + app_state["constant_columns"],
            errors="ignore",
        )
    )
    X_all = X_all.reindex(columns=feature_columns)
    probabilities = model.predict_proba(X_all)[:, 1]

    risk_df = df.copy()
    if app_state["excluded_id_columns"]:
        risk_df.insert(0, "Customer Reference", df[app_state["excluded_id_columns"][0]].values)
    else:
        risk_df.insert(0, "Customer Reference", np.arange(1, len(df) + 1))
    risk_df["Churn Probability"] = probabilities
    risk_df["Predicted Churn"] = np.where(probabilities >= threshold, "Yes", "No")
    risk_df["Risk Level"] = [assign_risk_level(probability, boundaries) for probability in probabilities]
    return risk_df.sort_values("Churn Probability", ascending=False)


def generate_recommendations(
    row: pd.Series,
    probability: float,
    threshold: float = 0.50,
    boundaries: dict[str, float] | None = None,
) -> list[str]:
    recommendations: list[str] = []
    lowered = {str(column).lower(): column for column in row.index}
    high_boundary = boundaries["medium_high"] if boundaries else 0.70

    def has_column_fragment(fragment: str) -> str | None:
        for lower_name, original in lowered.items():
            if fragment in lower_name:
                return original
        return None

    contract_col = has_column_fragment("contract")
    if contract_col and "month" in str(row[contract_col]).lower() and probability >= threshold:
        recommendations.append("Offer a long-term contract incentive or loyalty discount.")

    support_col = has_column_fragment("support")
    if support_col and str(row[support_col]).strip().lower() in {"no", "none", "false", "0"}:
        recommendations.append("Offer technical support or a service assistance package.")

    charges_col = has_column_fragment("charge") or has_column_fragment("bill")
    if charges_col and pd.to_numeric(pd.Series([row[charges_col]]), errors="coerce").notna().iloc[0]:
        recommendations.append("Review pricing and consider a targeted retention offer.")

    tenure_col = has_column_fragment("tenure") or has_column_fragment("subscriptionage")
    if tenure_col and pd.to_numeric(pd.Series([row[tenure_col]]), errors="coerce").fillna(999).iloc[0] <= 6:
        recommendations.append("Provide onboarding support or an early-stage loyalty incentive.")

    if not recommendations and probability >= high_boundary:
        recommendations.append("Prioritize this customer for a personalized retention outreach.")
    elif not recommendations:
        recommendations.append("Monitor customer engagement and maintain regular service follow-up.")

    return recommendations


def create_model_report(app_state: dict[str, Any], risk_counts: pd.Series | None = None) -> str:
    results = app_state["training_results"]
    best_name = results["best_model_name"]
    best_row = results["test_comparison"].set_index("Model").loc[best_name]

    buffer = StringIO()
    buffer.write("Predictive Customer Churn Analytics Model Report\n")
    buffer.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    buffer.write("Dataset Information\n")
    buffer.write(f"Records used for modeling: {len(app_state['y'])}\n")
    buffer.write(f"Target column: {app_state['target_col']}\n")
    buffer.write(f"Positive churn class: {app_state['positive_class']}\n")
    buffer.write(f"Churn rate: {app_state['y'].mean():.4f}\n")
    buffer.write(f"Identifier columns excluded: {', '.join(app_state['excluded_id_columns']) or 'None'}\n")
    buffer.write(f"Constant columns excluded: {', '.join(app_state['constant_columns']) or 'None'}\n\n")
    buffer.write("Preprocessing Approach\n")
    buffer.write("Numerical features: median imputation; scaled only for Logistic Regression.\n")
    buffer.write("Categorical features: most-frequent imputation and one-hot encoding with unknown-category handling.\n")
    buffer.write("All preprocessing is fitted inside scikit-learn pipelines after the train/validation/test split.\n\n")
    buffer.write("Models Tested\n")
    buffer.write("Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.\n")
    buffer.write("XGBoost is intentionally not mandatory for this implementation.\n\n")
    buffer.write("Best Model\n")
    buffer.write(f"{best_name}\n")
    buffer.write(f"Threshold: {results['thresholds'][best_name]:.2f}\n")
    buffer.write(f"Accuracy: {best_row['Accuracy']:.4f}\n")
    buffer.write(f"Precision: {best_row['Precision']:.4f}\n")
    buffer.write(f"Recall: {best_row['Recall']:.4f}\n")
    buffer.write(f"F1 Score: {best_row['F1 Score']:.4f}\n")
    buffer.write(f"ROC-AUC: {best_row['ROC-AUC']:.4f}\n")
    buffer.write(f"PR-AUC: {best_row['PR-AUC']:.4f}\n")
    buffer.write(f"Specificity: {best_row['Specificity']:.4f}\n\n")
    if risk_counts is not None:
        buffer.write("Customer Risk Summary\n")
        for risk_level, count in risk_counts.items():
            buffer.write(f"{risk_level}: {count}\n")
    return buffer.getvalue()


def reset_training_state() -> None:
    for key in [
        "training_results",
        "feature_columns",
        "excluded_id_columns",
        "constant_columns",
        "target_col",
        "positive_class",
        "y",
    ]:
        st.session_state.pop(key, None)


def sidebar_upload() -> None:
    st.sidebar.markdown("### Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        return

    file_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("file_signature") == file_signature:
        return

    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.sidebar.error("The uploaded CSV is empty.")
            return
        st.session_state.df = df
        st.session_state.file_signature = file_signature
        reset_training_state()
        st.sidebar.success("Dataset loaded successfully.")
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")


def render_home() -> None:
    st.markdown(
        """
<div class="hero">
    <h1>Predictive Customer Churn Analytics & Retention Decision Support System</h1>
    <p>An ML-powered system for predicting customer churn, identifying high-risk customers,
    explaining churn drivers, and supporting retention decisions.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("Upload Dataset -> Analyze -> Train Models -> Predict Risk -> Explain -> Take Action")
    st.info("Use the sidebar uploader to begin. The app will detect likely churn targets and keep XGBoost optional by using deployable scikit-learn models first.")


def render_dataset_page(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
    numeric_cols = convert_numeric_like_columns(df).select_dtypes(include=["number", "bool"]).columns
    categorical_count = df.shape[1] - len(numeric_cols)
    duplicate_rows = int(df.duplicated().sum())
    missing_values = int(df.isna().sum().sum())
    detected_target = detect_target_column(df)

    cols = st.columns(5)
    with cols[0]:
        display_metric_card("Rows", f"{len(df):,}")
    with cols[1]:
        display_metric_card("Columns", f"{df.shape[1]:,}")
    with cols[2]:
        display_metric_card("Numerical", f"{len(numeric_cols):,}")
    with cols[3]:
        display_metric_card("Categorical", f"{categorical_count:,}")
    with cols[4]:
        display_metric_card("Missing Values", f"{missing_values:,}")

    if detected_target:
        st.success(f"Detected target column: {detected_target}")
    else:
        st.warning("No obvious churn target was detected. Please choose the target column manually.")

    st.write(f"Duplicate rows detected: **{duplicate_rows:,}**")

    default_index = list(df.columns).index(detected_target) if detected_target in df.columns else 0
    target_col = st.selectbox("Select Target / Churn Column", df.columns, index=default_index)
    assessment = assess_target_column(df, target_col)

    for warning in assessment["warnings"]:
        st.warning(warning)
    for error in assessment["errors"]:
        st.error(error)

    if assessment["valid"]:
        target_counts = df[target_col].dropna().value_counts()
        st.markdown("Target class distribution")
        st.dataframe(
            pd.DataFrame(
                {
                    "Class": target_counts.index.astype(str),
                    "Count": target_counts.values,
                    "Share": (target_counts.values / target_counts.sum()),
                }
            ),
            use_container_width=True,
        )

        positive_default = infer_positive_class(df[target_col])
        unique_values = list(df[target_col].dropna().unique())
        positive_index = unique_values.index(positive_default) if positive_default in unique_values else 0
        positive_class = st.selectbox(
            "Which class represents churn?",
            unique_values,
            index=positive_index,
            help="The selected class is treated as the positive class for recall, precision, F1 and risk prediction.",
        )
        y, _ = normalize_target(df[target_col], positive_class)
        st.success(f"Positive class = {positive_class}; Negative class = all other target values.")
        st.pyplot(plot_churn_distribution(y))
        st.session_state.target_col = target_col
        st.session_state.positive_class = positive_class
    else:
        st.info("Training is disabled until a valid binary target column is selected.")

    st.markdown("<div class='section-title'>Dataset Preview</div>", unsafe_allow_html=True)
    preview_rows = st.radio("Rows to preview", [5, 10, 20], horizontal=True)
    st.dataframe(df.head(preview_rows), width="stretch")

    profile_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Missing Values": df.isna().sum().values,
            "Unique Values": df.nunique(dropna=True).values,
        }
    )
    st.dataframe(profile_df, width="stretch")

    id_columns = identify_id_columns(df, target_col)
    st.markdown("<div class='section-title'>Potential Identifier Columns</div>", unsafe_allow_html=True)
    if id_columns:
        id_df = pd.DataFrame(
            [{"Column": column, "Reason": reason} for column, reason in id_columns.items()]
        )
        st.dataframe(id_df, width="stretch")
        selected_ids = st.multiselect(
            "Columns to exclude from modeling",
            list(id_columns.keys()),
            default=list(id_columns.keys()),
        )
    else:
        st.info("No strong identifier columns detected.")
        selected_ids = []
    st.session_state.excluded_id_columns = selected_ids


def render_exploration_page(df: pd.DataFrame) -> None:
    if "target_col" not in st.session_state or "positive_class" not in st.session_state:
        st.warning("Please select a valid target on the Dataset page first.")
        return

    target_col = st.session_state.target_col
    y, target_mask = normalize_target(df[target_col], st.session_state.positive_class)
    analysis_df = df.loc[target_mask].copy()
    analysis_df["Churn Label"] = np.where(y.values == 1, "Churn", "Non-Churn")
    converted = convert_numeric_like_columns(
        analysis_df.drop(
            columns=[target_col] + st.session_state.get("excluded_id_columns", []),
            errors="ignore",
        )
    )

    st.markdown("<div class='section-title'>Data Exploration</div>", unsafe_allow_html=True)
    st.pyplot(plot_churn_distribution(y))

    numeric_cols = converted.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [column for column in converted.columns if column not in numeric_cols]

    col1, col2 = st.columns(2)
    with col1:
        if numeric_cols:
            selected_num = st.selectbox("Numerical feature vs churn", numeric_cols)
            plot_df = pd.DataFrame(
                {selected_num: converted[selected_num], "Churn Label": analysis_df["Churn Label"]}
            )
            fig, ax = plt.subplots(figsize=(5, 3.4), dpi=140)
            sns.boxplot(data=plot_df, x="Churn Label", y=selected_num, palette=["#38bdf8", "#ec4899"], ax=ax)
            ax.set_title(f"{selected_num} by Churn")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No numerical features are available for numerical exploration.")

    with col2:
        if categorical_cols:
            selected_cat = st.selectbox("Categorical feature vs churn", categorical_cols)
            cross_tab = pd.crosstab(converted[selected_cat], y, normalize="index") * 100
            if 1 in cross_tab.columns:
                churn_rate = cross_tab[1].sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(5, 3.4), dpi=140)
                sns.barplot(x=churn_rate.values, y=churn_rate.index.astype(str), color="#ec4899", ax=ax)
                ax.set_xlabel("Churn Rate (%)")
                ax.set_title(f"Churn Rate by {selected_cat}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No categorical features are available for categorical exploration.")

    st.markdown("<div class='section-title'>Outlier Analysis</div>", unsafe_allow_html=True)
    outliers = outlier_summary(converted)
    if outliers.empty:
        st.info("No major IQR-based outlier signals were detected in numerical columns.")
    else:
        outliers["Share"] = outliers["Share"].map(pct)
        st.dataframe(outliers, width="stretch")
        st.caption("Outliers are shown for review only. They are not automatically deleted.")


def render_training_page(df: pd.DataFrame) -> None:
    if "target_col" not in st.session_state or "positive_class" not in st.session_state:
        st.warning("Please select a valid target on the Dataset page first.")
        return

    st.markdown("<div class='section-title'>Model Training</div>", unsafe_allow_html=True)
    st.write("Models included: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.")
    st.caption("XGBoost is intentionally optional and is not required for this first deployable implementation.")

    manual_threshold = st.slider(
        "Prediction Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Lower thresholds catch more possible churners but may increase false positives.",
    )
    optimize = st.checkbox("Optimize threshold on validation data", value=True)
    threshold_objective = st.selectbox(
        "Threshold optimization objective",
        ["F1 Score", "Recall", "Balanced performance"],
    )
    selection_priority = st.selectbox(
        "Best model selection priority",
        ["F1 Score", "Recall", "ROC-AUC"],
        help="The best model is selected using validation performance, not the final test split.",
    )

    if st.button("Train Models", type="primary"):
        try:
            modeling_data = build_modeling_data(
                df,
                st.session_state.target_col,
                st.session_state.positive_class,
                st.session_state.get("excluded_id_columns", []),
            )
            X = modeling_data["X"]
            y = modeling_data["y"]

            if X.shape[1] == 0:
                st.error("No usable feature columns remain after excluding identifiers and constants.")
                return

            with st.spinner("Training models with leakage-safe preprocessing pipelines..."):
                results = train_models(
                    X,
                    y,
                    manual_threshold,
                    optimize,
                    threshold_objective,
                    selection_priority,
                )

            st.session_state.training_results = results
            st.session_state.feature_columns = X.columns.tolist()
            st.session_state.constant_columns = modeling_data["constant_columns"]
            st.session_state.y = y
            st.session_state.missing_numeric = modeling_data["missing_numeric"]
            st.session_state.missing_categorical = modeling_data["missing_categorical"]
            st.success("Training complete.")
        except Exception as exc:
            st.error(f"Model training could not be completed: {exc}")

    if "training_results" in st.session_state:
        results = st.session_state.training_results
        st.markdown("<div class='section-title'>Validation Model Comparison</div>", unsafe_allow_html=True)
        st.caption("This table is used for model selection, so the final test split stays untouched for selection decisions.")
        st.dataframe(format_metrics_table(results["validation_comparison"]), width="stretch")

        best_name = results["best_model_name"]
        best_validation = results["validation_comparison"].set_index("Model").loc[best_name]
        cols = st.columns(4)
        with cols[0]:
            display_metric_card("Best Model", best_name)
        with cols[1]:
            display_metric_card("Validation F1", pct(best_validation["F1 Score"]))
        with cols[2]:
            display_metric_card("Validation Recall", pct(best_validation["Recall"]))
        with cols[3]:
            display_metric_card("Chosen Threshold", f"{results['thresholds'][best_name]:.2f}")

        if results["failures"]:
            st.warning(f"Some models failed: {results['failures']}")


def render_evaluation_page() -> None:
    if "training_results" not in st.session_state:
        st.warning("Please train models first.")
        return

    results = st.session_state.training_results
    best_name = results["best_model_name"]
    y_test = results["splits"]["y_test"]
    y_prob = results["probabilities"][best_name]["test"]
    default_threshold = float(results["thresholds"][best_name])

    st.markdown("<div class='section-title'>Model Evaluation</div>", unsafe_allow_html=True)
    threshold = st.slider(
        "Evaluation threshold",
        min_value=0.10,
        max_value=0.90,
        value=default_threshold,
        step=0.01,
        help="This recomputes final metrics for the selected model. Optimized thresholds are learned only from validation data.",
    )
    metrics = calculate_metrics(y_test, y_prob, threshold)

    cols = st.columns(6)
    for column, metric_name in zip(
        cols,
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Specificity"],
    ):
        with column:
            display_metric_card(metric_name, pct(metrics[metric_name]))

    st.markdown("<div class='section-title'>Final Test Model Comparison</div>", unsafe_allow_html=True)
    st.caption("Reported from the untouched test split. The best model was selected earlier using validation metrics.")
    st.dataframe(format_metrics_table(results["test_comparison"]), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_confusion(y_test, y_prob, threshold))
    with col2:
        st.pyplot(plot_probability_distribution(y_prob, threshold))

    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(plot_roc_curves(results))
    with col4:
        st.pyplot(plot_pr_curve(y_test, y_prob))

    top_n = st.select_slider("Feature importance count", options=[5, 10, 15, 20], value=10)
    importance_df, method = get_feature_importance(
        results["models"][best_name],
        results["splits"]["X_test"],
        y_test,
        top_n,
    )
    st.markdown("<div class='section-title'>Top Churn Drivers</div>", unsafe_allow_html=True)
    st.caption(method)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    sns.barplot(data=importance_df, x="Importance", y="Feature", color="#ec4899", ax=ax)
    ax.set_title("Top Features")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download Feature Importance CSV",
        importance_df.to_csv(index=False),
        "feature_importance.csv",
        "text/csv",
    )


def render_risk_page(df: pd.DataFrame) -> None:
    if "training_results" not in st.session_state:
        st.warning("Please train models first.")
        return

    results = st.session_state.training_results
    best_name = results["best_model_name"]
    threshold = st.slider(
        "Risk prediction threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(results["thresholds"][best_name]),
        step=0.01,
    )

    risk_df = create_risk_table(df, st.session_state, threshold)
    risk_counts = risk_df["Risk Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)

    st.markdown("<div class='section-title'>Customer Risk Dashboard</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        display_metric_card("Total Customers", f"{len(risk_df):,}")
    with cols[1]:
        display_metric_card("High Risk", f"{risk_counts['High']:,}")
    with cols[2]:
        display_metric_card("Medium Risk", f"{risk_counts['Medium']:,}")
    with cols[3]:
        display_metric_card("Low Risk", f"{risk_counts['Low']:,}")

    selected_risk = st.multiselect(
        "Filter by risk level",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    filtered = risk_df[risk_df["Risk Level"].isin(selected_risk)]
    display_columns = [
        "Customer Reference",
        "Churn Probability",
        "Predicted Churn",
        "Risk Level",
    ] + [column for column in df.columns if column not in {"Customer Reference"}][:8]
    st.dataframe(filtered[display_columns], width="stretch")

    st.download_button(
        "Download Customer Risk Predictions CSV",
        risk_df.to_csv(index=False),
        "customer_risk_predictions.csv",
        "text/csv",
    )


def render_individual_prediction_page(df: pd.DataFrame) -> None:
    if "training_results" not in st.session_state:
        st.warning("Please train models first.")
        return

    results = st.session_state.training_results
    best_name = results["best_model_name"]
    model = results["models"][best_name]
    feature_columns = st.session_state.feature_columns
    X_source = convert_numeric_like_columns(
        df.drop(
            columns=[st.session_state.target_col]
            + st.session_state.get("excluded_id_columns", [])
            + st.session_state.get("constant_columns", []),
            errors="ignore",
        )
    ).reindex(columns=feature_columns)

    st.markdown("<div class='section-title'>Individual Customer Churn Predictor</div>", unsafe_allow_html=True)
    st.caption("Categorical choices are taken dynamically from the uploaded dataset.")

    with st.form("individual_prediction_form"):
        values: dict[str, Any] = {}
        columns = st.columns(2)
        for index, feature in enumerate(feature_columns):
            container = columns[index % 2]
            series = X_source[feature]
            with container:
                if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    default = float(numeric_series.median()) if numeric_series.notna().any() else 0.0
                    values[feature] = st.number_input(feature, value=default)
                else:
                    choices = series.dropna().astype(str).sort_values().unique().tolist()
                    if not choices:
                        choices = ["Unknown"]
                    values[feature] = st.selectbox(feature, choices)

        submitted = st.form_submit_button("Predict Churn", type="primary")

    if submitted:
        input_df = pd.DataFrame([values]).reindex(columns=feature_columns)
        probability = float(model.predict_proba(input_df)[:, 1][0])
        threshold = float(results["thresholds"][best_name])
        risk_level = assign_risk_level(probability)
        prediction = "Likely to churn" if probability >= threshold else "Not likely to churn"

        cols = st.columns(3)
        with cols[0]:
            display_metric_card("Churn Probability", pct(probability))
        with cols[1]:
            display_metric_card("Risk Level", risk_level)
        with cols[2]:
            display_metric_card("Prediction", prediction)

        st.markdown("<div class='section-title'>Suggested Retention Action</div>", unsafe_allow_html=True)
        for recommendation in generate_recommendations(pd.Series(values), probability):
            st.write(f"- {recommendation}")


def render_explainability_page() -> None:
    if "training_results" not in st.session_state:
        st.warning("Please train models first.")
        return

    results = st.session_state.training_results
    best_name = results["best_model_name"]
    st.markdown("<div class='section-title'>Explainable AI</div>", unsafe_allow_html=True)
    st.write(
        "This implementation uses model feature importance or logistic coefficients as a deployable explainability layer. "
        "If a model does not expose native importance values, the app falls back to permutation importance."
    )
    importance_df, method = get_feature_importance(
        results["models"][best_name],
        results["splits"]["X_test"],
        results["splits"]["y_test"],
        15,
    )
    st.caption(method)
    st.dataframe(importance_df, width="stretch")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    sns.barplot(data=importance_df, x="Importance", y="Feature", color="#f472b6", ax=ax)
    ax.set_title(f"Global Churn Drivers - {best_name}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_reports_page(df: pd.DataFrame) -> None:
    if "training_results" not in st.session_state:
        st.warning("Please train models first.")
        return

    results = st.session_state.training_results
    best_name = results["best_model_name"]
    threshold = float(results["thresholds"][best_name])
    risk_df = create_risk_table(df, st.session_state, threshold)
    risk_counts = risk_df["Risk Level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    importance_df, _ = get_feature_importance(
        results["models"][best_name],
        results["splits"]["X_test"],
        results["splits"]["y_test"],
        20,
    )

    st.markdown("<div class='section-title'>Reports</div>", unsafe_allow_html=True)
    st.download_button(
        "Download Model Comparison CSV",
        results["test_comparison"].to_csv(index=False),
        "model_comparison.csv",
        "text/csv",
    )
    st.download_button(
        "Download Feature Importance CSV",
        importance_df.to_csv(index=False),
        "feature_importance.csv",
        "text/csv",
    )
    st.download_button(
        "Download Customer Risk Report CSV",
        risk_df.to_csv(index=False),
        "customer_risk_predictions.csv",
        "text/csv",
    )
    st.download_button(
        "Download Model Performance Report",
        create_model_report(st.session_state, risk_counts),
        "model_report.txt",
        "text/plain",
    )


def render_preprocessing_page(df: pd.DataFrame) -> None:
    if "target_col" not in st.session_state or "positive_class" not in st.session_state:
        st.warning("Please select a valid target on the Dataset page first.")
        return

    modeling_data = build_modeling_data(
        df,
        st.session_state.target_col,
        st.session_state.positive_class,
        st.session_state.get("excluded_id_columns", []),
    )
    X = modeling_data["X"]
    numeric_features, categorical_features = split_feature_types(X)

    st.markdown("<div class='section-title'>Preprocessing Summary</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        display_metric_card("Model Features", f"{X.shape[1]:,}")
    with cols[1]:
        display_metric_card("Numerical Missing", f"{modeling_data['missing_numeric']:,}")
    with cols[2]:
        display_metric_card("Categorical Missing", f"{modeling_data['missing_categorical']:,}")
    with cols[3]:
        display_metric_card("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.write("Numerical preprocessing: median imputation; StandardScaler only for Logistic Regression.")
    st.write("Categorical preprocessing: most-frequent imputation and OneHotEncoder(handle_unknown='ignore').")
    st.write("Train/validation/test split happens before fitting any preprocessing step.")

    excluded = pd.DataFrame(
        {
            "Column": st.session_state.get("excluded_id_columns", []) + modeling_data["constant_columns"],
            "Reason": ["Identifier excluded"] * len(st.session_state.get("excluded_id_columns", []))
            + ["Constant feature excluded"] * len(modeling_data["constant_columns"]),
        }
    )
    if excluded.empty:
        st.info("No identifier or constant columns are currently excluded.")
    else:
        st.dataframe(excluded, width="stretch")

    feature_profile = pd.DataFrame(
        {
            "Feature Type": ["Numerical", "Categorical"],
            "Count": [len(numeric_features), len(categorical_features)],
        }
    )
    st.dataframe(feature_profile, width="stretch")


def main() -> None:
    configure_page()
    sidebar_upload()

    page_options = [
        "Overview",
        "Dataset",
        "Data Exploration",
        "Preprocessing",
        "Model Training",
        "Model Evaluation",
        "Customer Risk",
        "Individual Prediction",
        "Explainable AI",
        "Reports",
    ]
    page = st.sidebar.radio("Navigation", page_options)

    df = st.session_state.get("df")
    if page == "Overview":
        render_home()
        return

    if df is None:
        st.warning("Please upload a dataset first.")
        return

    if page == "Dataset":
        render_dataset_page(df)
    elif page == "Data Exploration":
        render_exploration_page(df)
    elif page == "Preprocessing":
        render_preprocessing_page(df)
    elif page == "Model Training":
        render_training_page(df)
    elif page == "Model Evaluation":
        render_evaluation_page()
    elif page == "Customer Risk":
        render_risk_page(df)
    elif page == "Individual Prediction":
        render_individual_prediction_page(df)
    elif page == "Explainable AI":
        render_explainability_page()
    elif page == "Reports":
        render_reports_page(df)


if __name__ == "__main__":
    main()
