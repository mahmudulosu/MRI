# -*- coding: utf-8 -*-
"""
Leakage-safe PCA explained variance curve (median + IQR across folds)

- Resample all MRIs onto a shared 2mm reference grid
- Compute a group brain mask
- Vectorize masked voxels
- In each CV fold: center TRAIN only and fit PCA on TRAIN only
- Plot median cumulative explained variance with IQR shading
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.image import resample_img, resample_to_img
from nilearn.masking import compute_brain_mask
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def is_nii(fp: str) -> bool:
    f = fp.lower()
    return f.endswith(".nii") or f.endswith(".nii.gz")


def list_nifti_files(folder: str):
    return [os.path.join(folder, f) for f in os.listdir(folder) if is_nii(os.path.join(folder, f))]


def subject_id_from_path(fp: str) -> str:
    return os.path.basename(fp).split("_")[0]


def make_reference_2mm(img: nib.Nifti1Image) -> nib.Nifti1Image:
    target_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    return resample_img(img, target_affine=target_affine, interpolation="continuous")


def load_resample_and_mask(file_paths, ref_img_2mm):
    data_list, mask_list = [], []
    for fp in file_paths:
        img = nib.load(fp)
        img_2mm = resample_to_img(img, ref_img_2mm, interpolation="continuous")
        mask_2mm = compute_brain_mask(img_2mm)
        data_list.append(img_2mm.get_fdata().astype(np.float32))
        mask_list.append(mask_2mm.get_fdata().astype(bool))
    return data_list, mask_list


def make_group_mask(mask_list, min_prop: float = 1.0) -> np.ndarray:
    m = np.stack(mask_list, axis=0)
    thresh = int(np.ceil(min_prop * m.shape[0]))
    return (m.sum(axis=0) >= thresh)


def vectorize_with_mask(data_list, group_mask):
    idx = np.where(group_mask.ravel())[0]
    X = np.vstack([d.ravel()[idx] for d in data_list]).astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    return X


def foldwise_variance_curve(X, y, n_splits=5, seed=1337, max_components=300, elbow_delta=0.005):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    curves = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr = X[tr]

        scaler = StandardScaler(with_mean=True, with_std=False)
        Xtr_c = scaler.fit_transform(Xtr)

        m = min(max_components, Xtr_c.shape[0], Xtr_c.shape[1])
        pca = PCA(n_components=m, svd_solver="randomized", random_state=seed)
        pca.fit(Xtr_c)

        cum = np.cumsum(pca.explained_variance_ratio_)
        curves.append(cum)
        print(f"[Fold {fold}] curve length={len(cum)}  var@K=50={cum[min(49, len(cum)-1)]*100:.1f}%")

    L = min(len(c) for c in curves)
    M = np.vstack([c[:L] for c in curves])

    median = np.median(M, axis=0)
    q1 = np.percentile(M, 25, axis=0)
    q3 = np.percentile(M, 75, axis=0)

    marginal = np.diff(np.r_[0, median])
    elbow = int(np.argmax(marginal < elbow_delta))
    elbow = max(elbow, 1)

    ks = np.arange(1, L + 1)
    return ks, median, q1, q3, elbow


def plot_variance_curve(ks, median, q1, q3, mark_k=50, elbow_k=None, save_path="pca_variance_cv.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(ks, median, label="Median (train-only CV)")
    plt.fill_between(ks, q1, q3, alpha=0.2, label="IQR")
    if mark_k <= ks[-1]:
        plt.axvline(mark_k, ls="--", lw=1)
        plt.axhline(median[mark_k - 1], ls="--", lw=1)
    if elbow_k is not None and elbow_k <= ks[-1]:
        plt.axvline(elbow_k, ls=":", lw=1, label=f"Elbow ≈ {elbow_k}")
    plt.xlabel("Number of components (K)")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance (leakage-safe)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def main():
    csv_path = r"C:\research\MRI\participants_LSD_andLEMON.csv"
    data_folder = r"C:\research\commonMRI"

    df = pd.read_csv(csv_path)
    if "participant_id" not in df.columns or "gender" not in df.columns:
        raise RuntimeError("CSV must contain participant_id and gender columns for this script.")

    # Encode gender: female=0, male=1
    def encode_gender(val):
        s = str(val).strip().lower()
        if s in {"f", "female", "0"}:
            return 0
        if s in {"m", "male", "1"}:
            return 1
        return None

    df["y"] = df["gender"].map(encode_gender)
    df = df.dropna(subset=["y"])
    pid_to_y = dict(zip(df["participant_id"].astype(str), df["y"].astype(int)))

    all_files = sorted(list_nifti_files(data_folder))
    file_paths, labels, seen = [], [], set()

    for fp in all_files:
        pid = subject_id_from_path(fp)
        if pid in pid_to_y and pid not in seen:
            file_paths.append(fp)
            labels.append(pid_to_y[pid])
            seen.add(pid)

    y = np.array(labels, dtype=int)
    print(f"Subjects: {len(file_paths)} (female={(y==0).sum()}, male={(y==1).sum()})")

    ref_img_2mm = make_reference_2mm(nib.load(file_paths[0]))
    data_list, mask_list = load_resample_and_mask(file_paths, ref_img_2mm)

    group_mask = make_group_mask(mask_list, min_prop=1.0)
    print("Group mask voxels:", int(group_mask.sum()))

    X = vectorize_with_mask(data_list, group_mask)
    print("X shape:", X.shape)

    ks, median, q1, q3, elbow_k = foldwise_variance_curve(X, y, n_splits=5)
    plot_variance_curve(ks, median, q1, q3, mark_k=50, elbow_k=elbow_k)


if __name__ == "__main__":
    main()
