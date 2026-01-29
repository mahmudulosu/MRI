import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.image import smooth_img
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.maskers import NiftiLabelsMasker

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


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


def load_and_preprocess_mri(fp, fwhm=8.0):
    return smooth_img(nib.load(fp), fwhm=fwhm)


def multiclass_roc_auc(y_true, y_proba):
    n_classes = len(CLASS_NAMES)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    aucs = {}
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
        aucs[CLASS_NAMES[c]] = auc(fpr, tpr)
    return aucs


def main():
    parser = argparse.ArgumentParser(description="4-class age×gender (ROI features + linear SVM).")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--fwhm", type=float, default=8.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = pd.read_csv(args.csv_path)
    pid_to_y = build_4class_labels(meta)
    print("Subjects with 4-class labels:", len(pid_to_y))

    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True, memory="nilearn_cache", verbose=0)

    files = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder)
             if f.endswith(".nii") or f.endswith(".nii.gz")]

    X_list, y_list = [], []
    for fp in files:
        pid = os.path.basename(fp).split("_")[0]
        if pid not in pid_to_y:
            continue
        try:
            img = load_and_preprocess_mri(fp, fwhm=args.fwhm)
            feats = masker.fit_transform(img).ravel()
            X_list.append(feats)
            y_list.append(pid_to_y[pid])
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=int)

    if len(X) == 0:
        raise SystemExit("No matched subjects found. Check folder and CSV IDs.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel="linear", probability=True, random_state=args.seed)
    clf.fit(X_train_s, y_train)

    pred = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)

    acc = accuracy_score(y_test, pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASS_NAMES))))
    print("Confusion matrix:\n", cm)

    aucs = multiclass_roc_auc(y_test, proba)
    print("One-vs-rest AUCs:", aucs)


if __name__ == "__main__":
    main()
