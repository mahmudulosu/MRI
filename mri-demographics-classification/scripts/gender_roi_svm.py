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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


def load_and_preprocess_mri(file_path: str, fwhm: float = 8.0):
    img = nib.load(file_path)
    return smooth_img(img, fwhm=fwhm)


def extract_roi_features(img, masker: NiftiLabelsMasker) -> np.ndarray:
    return masker.fit_transform(img).ravel()


def plot_confusion(cm, classes, title):
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def plot_roc(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return roc_auc


def main():
    parser = argparse.ArgumentParser(description="Gender classification (ROI features + linear SVM).")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--fwhm", type=float, default=8.0)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = pd.read_csv(args.csv_path)
    meta["gender"] = meta["gender"].astype(str).str.upper().str.strip()
    meta["participant_id"] = meta["participant_id"].astype(str)

    label_map = {"F": 0, "M": 1}
    pid_to_y = {r["participant_id"]: label_map[r["gender"]]
                for _, r in meta.iterrows() if r["gender"] in label_map}

    # Atlas
    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    labels = list(atlas.labels)
    feature_names = labels[1:] if labels and str(labels[0]).lower().startswith("background") else labels

    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True, memory="nilearn_cache", verbose=0)

    # Files
    nii_files = [os.path.join(args.data_folder, f) for f in os.listdir(args.data_folder)
                 if f.endswith(".nii") or f.endswith(".nii.gz")]

    X_list, y_list = [], []
    for fp in nii_files:
        pid = os.path.basename(fp).split("_")[0]
        if pid not in pid_to_y:
            continue
        try:
            img = load_and_preprocess_mri(fp, fwhm=args.fwhm)
            feats = extract_roi_features(img, masker)
            X_list.append(feats)
            y_list.append(pid_to_y[pid])
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=int)

    if len(X) == 0:
        raise SystemExit("No subjects matched CSV labels. Check filenames and participant_id.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel="linear", probability=True, random_state=args.seed)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, ["Female", "Male"], "Confusion Matrix (Gender, ROI-SVM)")

    y_score = clf.predict_proba(X_test_s)[:, 1]
    roc_auc = plot_roc(y_test, y_score, "ROC (Gender, ROI-SVM)")
    print("ROC AUC:", roc_auc)


if __name__ == "__main__":
    main()
