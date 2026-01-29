import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from scipy.ndimage import binary_erosion
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE


CLASS_NAMES = ["young_M", "young_F", "old_M", "old_F"]
VALID_AGES = ["20-25", "25-30", "60-65", "65-70", "70-75"]
AGE_TO_GROUP = {"20-25": "young", "25-30": "young", "60-65": "old", "65-70": "old", "70-75": "old"}


def build_4class_labels(meta: pd.DataFrame):
    meta = meta.copy()
    meta["participant_id"] = meta["participant_id"].astype(str)
    meta["age_str"] = meta["age"].astype(str).str.strip()
    meta["gender_str"] = meta["gender"].astype(str).str.strip().str.upper()
    meta = meta[meta["age_str"].isin(VALID_AGES) & meta["gender_str"].isin(["M", "F"])]

    pid_to_y = {}
    for _, r in meta.iterrows():
        pid = r["participant_id"]
        age_group = AGE_TO_GROUP[r["age_str"]]
        g = r["gender_str"]
        if age_group == "young" and g == "M":
            pid_to_y[pid] = 0
        elif age_group == "young" and g == "F":
            pid_to_y[pid] = 1
        elif age_group == "old" and g == "M":
            pid_to_y[pid] = 2
        elif age_group == "old" and g == "F":
            pid_to_y[pid] = 3
    return pid_to_y


def extract_morphometric(fp: str) -> np.ndarray:
    img = nib.load(fp)
    data = img.get_fdata()

    brain = data > 0
    total = float(np.sum(brain))

    eroded = binary_erosion(brain, iterations=2)
    cortical_shell = brain & ~eroded
    cortical_prop = (np.sum(cortical_shell) / total) if total > 0 else 0.0

    x = brain.shape[0]
    left = np.sum(eroded[: x // 2, :, :])
    right = np.sum(eroded[x // 2 :, :, :])

    return np.array([cortical_prop, left, right, total], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="4-class age×gender (morphometric features + SVM + SMOTE + GridSearch).")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = pd.read_csv(args.csv_path)
    pid_to_y = build_4class_labels(meta)
    print("Subjects with 4-class labels:", len(pid_to_y))

    files = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder)
             if f.endswith(".nii") or f.endswith(".nii.gz")]

    X_list, y_list = [], []
    for fp in files:
        pid = os.path.basename(fp).split("_")[0]
        if pid not in pid_to_y:
            continue
        try:
            X_list.append(extract_morphometric(fp))
            y_list.append(pid_to_y[pid])
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=int)

    if len(X) == 0:
        raise SystemExit("No matched subjects found for 4-class morphometric classification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # SMOTE on training only
    sm = SMOTE(random_state=args.seed)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_res)
    X_test_s = scaler.transform(X_test)

    param_grid = {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["linear", "rbf"]}
    base = SVC(probability=True, class_weight="balanced", random_state=args.seed)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    gs = GridSearchCV(base, param_grid, cv=cv, scoring="accuracy")
    gs.fit(X_train_s, y_train_res)

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    pred = best.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))


if __name__ == "__main__":
    main()
