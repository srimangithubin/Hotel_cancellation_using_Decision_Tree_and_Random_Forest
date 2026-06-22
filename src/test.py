import joblib 
import pandas as pd

from train import(
    DATA_PATH,
    MODEL_DIR,
    TARGET,
    load_data,
    clean_and_engineer_features
)

def main():
    model_path = MODEL_DIR / 'best_model.joblib'

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}."
                                "Run python src/train.py first. "
                                )

    print("Loading saved model...")
    model = joblib.load(model_path)

    estimator = model.named_steps.get('model')
    if hasattr(estimator, 'n_jobs'):
        estimator.set_params(n_jobs=1)

    print("Loading and preparing data...")
    df = load_data(DATA_PATH)
    df_model = clean_and_engineer_features(df)

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET].astype(int)

    sample = X.iloc[[0]]
    actual_value = y.iloc[0]

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    print("\n===== Single Prediction Test =====")
    print("Actual value:", actual_value)
    print("Predicted value:", prediction)
    print("Cancellation probability:", round(float(probability), 4))

    print("\nSample input used:")
    print(sample.T)

if __name__ == "__main__":
    main() 
