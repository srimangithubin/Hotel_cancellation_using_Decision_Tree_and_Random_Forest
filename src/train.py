from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.inspection import permutation_importance

# =========================
# Project configuration
# =========================

RANDOM_STATE = 42
TARGET = 'is_canceled'

PROJECT_DIR = Path(__file__).resolve().parent[1]

DATA_PATH = PROJECT_DIR / 'data' / 'raw' / 'hotels.csv'

MODEL_DIR = PROJECT_DIR / 'models'
REPORT_DIR = PROJECT_DIR / 'reports'
IMAGE_DIR = PROJECT_DIR / 'images'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)



# =========================
# Load data
# =========================


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Data file not found at {path} '
                                "Place hotels.csv inside data/raw/ before running this script."
                                )
    
    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")

    )

    return df


# =========================
# Cleaning and feature engineering
# =========================

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # remove leakage columns
    # These columns reveal final outcome information

    leakage_columns = [
        'reservation_status',
        'reservation_status_date'
    ]

    df = df.drop(
        columns = [col for col in leakage_columns if col in df.columns]
    )

    # craete useful binary indicators before dropping ID-like columns

    if 'agent' in df.columns:
        df['has_agent'] = df['agent'].notna().astype(int)

    if 'company' in df.columns:
        df['has_company'] = df['company'].notna().astype(int)

    # Drop high-cardinality ID-like columns
    
    id_like_columns = ['agent', 'company']
    
    df = df.drop(
        columns = [col for col in id_like_columns if col in df.columns]
    )

    # Handle children missing values before creating total guests.

    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0)

    # Total stay duration
    if 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

    # total number of guests

    if {'adults', 'children', 'babies'}.issubset(df.columns):
        df['total_guests'] = df['adults'] + df['children'] + df['babies']

        df['has_children'] = (
            (df['children'] + df['babies'] > 0)
        ).astype(int)

    # Removing  impossible bookings with zero guests

    if 'total_guests' in df.columns:
        df = df[df['total_guests'] > 0]

    # Remove unrealistic negative room rates
    if 'adr' in df.columns:
        df = df[df['adr']>=0]

    return df


# =========================
# Split features and target
# =========================

def split_features_target(df: pd.dataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    return X,y


# =========================
# Preprocessing
# =========================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    numeric_pipeline = Pipeline(
        steps =[
            ('imputer', SimpleImputer(strategy='median'))
                ]
    )

    categorical_pipeline = Pipeline(
        steps =[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


# =========================
# Build pipelines
# =========================


def build_pipeline(preprocessor, model) -> Pipeline:
    pipeline = Pipeline(
        steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )

    return pipeline

# =========================
# Cross-validation
# =========================

def run_cross_validation(model_name, pipeline, X_train, y_train, cv):
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv = cv,
        scoring = scoring,
        return_train_score = True
    )

    result = {
        "model": model_name,
        "cv_accuracy": scores["test_accuracy"].mean(),
        "cv_precision": scores["test_precision"].mean(),
        "cv_recall": scores["test_recall"].mean(),
        "cv_f1": scores["test_f1"].mean(),
        "cv_roc_auc": scores["test_roc_auc"].mean(),
        "train_f1": scores["train_f1"].mean(),
        "f1_std": scores["test_f1"].std()
    }

    return result

# =========================
# Test-set evaluation
# =========================

def evaluate_model(model_name, fitted_model, X_test, y_test):
    y_pred = fitted_model.predict(X_test)
    y_proba = fitted_model.predict_proba(X_test)[:,1]

    results = {
        "model": model_name,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba)
    }

    print (f'\n === {model_name} Test Results ===')
    print(json.dumps(results, indent=4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix = cm)
    disp.plot()
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()

    cm_path = IMAGE_DIR / f'{model_name.lower().replace(" ","_")}_confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()

    # Roc Curve
    RocCurveDisplay.from_estimators(fitted_model, X_test, y_test)
    plt.title(f'{model_name} ROC curve')
    plt.tight_layout()

    roc_path = IMAGE_DIR / f'{model_name.lower().replace(" ","_")}_roc_curve.png'
    plt.savefig(roc_path)
    plt.close()
    
    return results

# =========================
# Feature importance
# =========================

def save_feaeture_importance(best_model, X_test, y_test):
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats = 10,
        random_state = RANDOM_STATE,
        scoring = 'f1',
        n_jobs = -1
    )

    importance_df = pd.DataFrame(
        {
            'feature': X_test.columns,
            'importance_mean': importance.importances_mean,
            'importance_std': importance.importances_std
        }
    ).sort_values(by = 'importance_mean', ascending = False)

    importance_path = REPORT_DIR / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)

    top_features = importance_df.head(15)

    plt.figure(figsize=(10,6))
    plt.barh(top_features['feature'], top_features['importance_mean'])
    plt.gca().invert_yaxis()
    plt.title('Top 15 feature importance (permutation importance)')
    plt.xlabel('Mean F1 importance')
    plt.tight_layout()

    image_path = IMAGE_DIR / 'feature_importance.png'
    plt.savefig(image_path)
    plt.close()

    return importance_df

# =========================
# Main training workflow
# =========================

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning and engineering features...")
    df_model = clean_and_engineer_features(df)

    print("Dataset shape after cleaning:", df_model.shape)

    print('\n Target distribution:')
    print(df_model['target'].value_counts())

    X,y = split_features_target(df_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = RANDOM_STATE,
        stratify = y
    )

    preprocessor = build_preprocessor(X_train)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    baseline_model = build_pipeline(
        preprocessor,
        DummyClassifier(strategy='most_frequent')
     )
    
    decision_tree_model = build_pipeline(
        preprocessor,
        DecisionTreeClassifier(random_state=RANDOM_STATE,
                               class_weight='balanced',
                               n_jobs=-1)
    )

    random_forest_model = build_pipeline(
        preprocessor,
        RandomForestClassifier(random_state=RANDOM_STATE,
                               class_weight='balanced',
                               n_jobs=-1)
    )

    print("\nRunning cross-validation...")
    cv_results = []

    cv_results.append(
        run_cross_validation("Baseline Dummy Classifier", baseline_model, X_train, y_train, cv)
    )

    cv_results.append(
        run_cross_validation("Decision Tree", decision_tree_model, X_train, y_train, cv)
    )

    cv_results.append(
        run_cross_validation("Random Forest", random_forest_model, X_train, y_train, cv)
    )

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_path = REPORT_DIR / 'cv_results_before_tuning.csv'
    cv_results_df.to_csv(cv_results_path, index=False)

    print("\nCross-validation results:")
    print(cv_results_df)

    print("\nTuning Decision Tree:")
    dt_param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [4, 6, 8, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10, 20, 50],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
        "model__max_features": [None, "sqrt", "log2"]
    }

    dt_search = RandomizedSearchCV(


