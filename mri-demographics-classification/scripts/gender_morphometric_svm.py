import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from scipy.ndimage import binary_erosion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def extract_morphometric_features(file_path: str) -> np.ndarray:
    img = nib.load(file_path)
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
    parser = argparse.ArgumentParser(description="Gender classification (morphometric features + SVM).")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = pd.read_csv(args.csv_path)
    meta["gender"] = meta["gender"].astype(str).str.upper().str.strip()
    meta["participant_id"] = meta["participant_id"].astype(str)

    label_map = {"F": 0, "M": 1}
    pid_to_y = {r["participant_id"]: label_map[r["gender"]]
                for _, r in meta.iterrows() if r["gender"] in label_map}

    files = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder)
             if f.endswith(".nii") or f.endswith(".nii.gz")]

    X_list, y_list = [], []
    for fp in files:
        pid = os.path.basename(fp).split("_")[0]
        if pid not in pid_to_y:
            continue
        try:
            X_list.append(extract_morphometric_features(fp))
            y_list.append(pid_to_y[pid])
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=int)

    if len(X) == 0:
        raise SystemExit("No matched MRI files found.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel="linear", probability=True, random_state=args.seed)
    clf.fit(X_train_s, y_train)

    pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, pred, target_names=["Female", "Male"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))


if __name__ == "__main__":
    main()
