import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


VALID_AGES = ["20-25", "25-30", "60-65", "65-70", "70-75"]
AGE_GROUP_MAPPING = {
    "20-25": "young",
    "25-30": "young",
    "60-65": "old",
    "65-70": "old",
    "70-75": "old",
}
LABEL_MAPPING = {"young": 0, "old": 1}


def load_and_merge(mri_feature_path: str, age_csv_path: str) -> pd.DataFrame:
    mri_features = pd.read_csv(mri_feature_path)
    age_data = pd.read_csv(age_csv_path)

    # Basic checks
    required_mri_cols = {"participant_id", "WM_voxels", "GM_voxels", "CSF_voxels"}
    required_age_cols = {"participant_id", "age"}

    missing_mri = required_mri_cols - set(mri_features.columns)
    missing_age = required_age_cols - set(age_data.columns)

    if missing_mri:
        raise ValueError(f"Missing columns in MRI feature CSV: {sorted(missing_mri)}")
    if missing_age:
        raise ValueError(f"Missing columns in age CSV: {sorted(missing_age)}")

    # Filter and map ages
    age_data = age_data[age_data["age"].isin(VALID_AGES)].copy()
    age_data["age_group"] = age_data["age"].map(AGE_GROUP_MAPPING)
    age_data["label"] = age_data["age_group"].map(LABEL_MAPPING)

    # Merge
    merged = pd.merge(mri_features, age_data, on="participant_id", how="inner")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Random Forest on WM/GM/CSF voxel counts (young vs old).")
    parser.add_argument("--mri_feature_path", type=str, required=True, help="Path to mri_voxel_features.csv")
    parser.add_argument("--age_csv_path", type=str, required=True, help="Path to participants CSV")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--outputs_dir", type=str, default="outputs")

    args = parser.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)

    merged_data = load_and_merge(args.mri_feature_path, args.age_csv_path)

    if merged_data.empty:
        raise SystemExit("❌ No matching MRI and age data found. Ensure participant IDs match.")

    X = merged_data[["WM_voxels", "GM_voxels", "CSF_voxels"]]
    y = merged_data["label"]

    # Stratify to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        max_depth=args.max_depth,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Classification Accuracy: {acc:.3f}")
    print("✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=["young", "old"]))

    # Save outputs
    model_path = os.path.join(args.outputs_dir, "rf_voxel_counts.joblib")
    joblib.dump(clf, model_path)

    metrics = {
        "accuracy": float(acc),
        "n_samples_total": int(len(merged_data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": ["WM_voxels", "GM_voxels", "CSF_voxels"],
        "label_mapping": LABEL_MAPPING,
        "age_bins_used": VALID_AGES,
    }
    metrics_path = os.path.join(args.outputs_dir, "rf_voxel_counts_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
