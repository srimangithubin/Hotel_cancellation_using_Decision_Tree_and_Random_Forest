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

Random_STATE = 42
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
