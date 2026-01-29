import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.image import resample_img
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


def load_and_flatten(fp: str) -> np.ndarray:
    img = nib.load(fp)
    img2 = resample_img(img, target_affine=np.eye(3) * 2.0, interpolation="continuous")
    return img2.get_fdata().astype(np.float32).ravel()


def main():
    parser = argparse.ArgumentParser(description="4-class age×gender (PCA voxel features + linear SVM).")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = pd.read_csv(args.csv_path)
    pid_to_y = build_4class_labels(meta)

    files = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder)
             if f.endswith(".nii") or f.endswith(".nii.gz")]
    files.sort()

    X_all = np.stack([load_and_flatten(fp) for fp in files], axis=0)
    pca = PCA(n_components=args.n_components, random_state=args.seed)
    X_pca = pca.fit_transform(X_all)

    X, y = [], []
    for i, fp in enumerate(files):
        pid = os.path.basename(fp).split("_")[0]
        if pid in pid_to_y:
            X.append(X_pca[i])
            y.append(pid_to_y[pid])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    if len(X) == 0:
        raise SystemExit("No matched subjects found for 4-class labels.")

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
    print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))


if __name__ == "__main__":
    main()
