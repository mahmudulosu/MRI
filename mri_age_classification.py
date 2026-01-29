"""
MRI Age-Group Classification (PCA + Random Forest, 10-Fold CV)

What this script does:
1) Reads a participants CSV containing: participant_id, age
2) Maps ages into two classes: young vs old
3) Loads NIfTI MRIs from a folder, matches them to participant_id by filename prefix
4) Resamples all MRIs onto a single common grid (same shape/affine), then flattens to vectors
5) Runs 10-fold cross-validation with a leakage-free sklearn Pipeline:
      StandardScaler -> PCA -> RandomForestClassifier
6) Prints fold accuracies + mean/std accuracy

Filename assumption:
- Participant ID is extracted as: basename(file).split('_')[0]
  Example: "sub-001_T1w.nii.gz" -> "sub-001"
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.image import resample_img, resample_to_img

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Data / Label Utilities
# -----------------------------
def build_age_mapping(
    csv_path: str,
    participant_col: str = "participant_id",
    age_col: str = "age",
):
    """
    Load CSV and map participants into binary age groups.

    Valid ages:
        20-25, 25-30 -> young
        60-65, 65-70, 70-75 -> old
    """
    age_data = pd.read_csv(csv_path)

    if participant_col not in age_data.columns or age_col not in age_data.columns:
        raise ValueError(
            f"CSV must contain columns '{participant_col}' and '{age_col}'. "
            f"Found columns: {list(age_data.columns)}"
        )

    valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
    age_group_mapping = {
        "20-25": "young",
        "25-30": "young",
        "60-65": "old",
        "65-70": "old",
        "70-75": "old",
    }

    filtered = age_data[age_data[age_col].isin(valid_ages)].copy()

    age_mapping = {
        str(row[participant_col]): age_group_mapping[row[age_col]]
        for _, row in filtered.iterrows()
    }

    return age_mapping, age_data


def list_nifti_files(folder: str):
    """Return sorted list of .nii/.nii.gz paths in a folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    files = []
    for p in folder_path.iterdir():
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            files.append(str(p))
    files.sort()
    return files


def match_files_to_labels(file_paths, age_mapping):
    """
    Match MRI file paths to labels based on participant_id extracted from filename prefix.
    Returns:
      matched_paths, labels_str
    """
    matched_paths = []
    labels = []

    for fp in file_paths:
        subject_id = os.path.basename(fp).split("_")[0]
        if subject_id in age_mapping:
            matched_paths.append(fp)
            labels.append(age_mapping[subject_id])

    return matched_paths, np.array(labels)


# -----------------------------
# MRI Loading / Preprocessing
# -----------------------------
def make_reference_image(first_file: str, voxel_size_mm: float = 2.0):
    """
    Create a reference image defining the target grid.
    We resample the first image to an isotropic target_affine and use it as reference.
    """
    img = nib.load(first_file)

    target_affine = np.eye(3) * float(voxel_size_mm)
    ref_img = resample_img(img, target_affine=target_affine, interpolation="continuous")
    return ref_img


def load_resample_flatten(file_path: str, ref_img):
    """
    Load MRI, resample it onto the reference image grid, and flatten to 1D vector.
    """
    img = nib.load(file_path)
    img_rs = resample_to_img(img, ref_img, interpolation="continuous")
    data = img_rs.get_fdata().astype(np.float32).ravel()
    return data


def load_dataset_matrix(matched_paths, voxel_size_mm: float = 2.0):
    """
    Load all matched MRIs, resample them to a common grid, and build X matrix.
    """
    if len(matched_paths) == 0:
        raise ValueError("No MRI files matched to labels. Check filenames vs CSV participant_id.")

    ref_img = make_reference_image(matched_paths[0], voxel_size_mm=voxel_size_mm)

    vectors = []
    for i, fp in enumerate(matched_paths, start=1):
        vec = load_resample_flatten(fp, ref_img)
        vectors.append(vec)
        if i % 10 == 0 or i == len(matched_paths):
            print(f"Loaded & processed {i}/{len(matched_paths)} MRIs")

    X = np.stack(vectors, axis=0)
    return X


# -----------------------------
# Modeling / Evaluation
# -----------------------------
def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    shuffle: bool = True,
    random_state: int = 42,
    n_components: int = 50,
    n_estimators: int = 100,
    stratified: bool = False,
):
    """
    Leakage-free CV using an sklearn Pipeline (scaler + PCA + RF).
    PCA is fit ONLY on each training fold.
    """
    if stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
            ("rf", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
        ]
    )

    fold_accuracies = []
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y if stratified else None), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        all_true.append(y_test)
        all_pred.append(y_pred)

        print(f"Fold {fold:02d} Accuracy: {acc * 100:.2f}%")

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    print("\n==============================")
    print(f"Mean Accuracy: {mean_acc * 100:.2f}%")
    print(f"Std  Accuracy: {std_acc * 100:.2f}%")
    print("==============================\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_true, all_pred))
    print("\nClassification Report:")
    print(classification_report(all_true, all_pred, target_names=["young", "old"]))

    return fold_accuracies, mean_acc, std_acc


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="MRI Age-Group Classification (PCA + RF, 10-fold CV)")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=r"C:\research\MRI\participants_LSD_andLEMON.csv",
        help="Path to participants CSV containing participant_id and age columns",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=r"C:\research\common_mri",
        help="Folder containing NIfTI MRI files (.nii / .nii.gz)",
    )
    parser.add_argument("--participant_col", type=str, default="participant_id")
    parser.add_argument("--age_col", type=str, default="age")

    parser.add_argument("--voxel_size_mm", type=float, default=2.0, help="Target isotropic voxel size (mm)")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--n_estimators", type=int, default=100, help="RandomForest n_estimators")

    parser.add_argument("--n_splits", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before splitting (recommended)")
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use StratifiedKFold (recommended if classes are imbalanced)",
    )

    args = parser.parse_args()

    # 1) Load CSV + build age mapping
    age_mapping, full_csv = build_age_mapping(
        args.csv_path, participant_col=args.participant_col, age_col=args.age_col
    )
    print("Columns in CSV:", list(full_csv.columns))
    print(f"Participants with valid age labels: {len(age_mapping)}")

    # 2) Find MRI files
    file_paths = list_nifti_files(args.data_folder)
    print(f"Total NIfTI files found: {len(file_paths)}")

    # 3) Match MRIs to labels
    matched_paths, labels_str = match_files_to_labels(file_paths, age_mapping)
    print(f"Matched MRI files to labels: {len(matched_paths)}")

    if len(matched_paths) == 0:
        raise SystemExit(
            "No MRI files matched to CSV participant IDs.\n"
            "Check that filenames begin with participant_id followed by an underscore."
        )

    # 4) Encode labels
    label_mapping = {"young": 0, "old": 1}
    y = np.array([label_mapping[lbl] for lbl in labels_str], dtype=int)

    # 5) Load dataset matrix (resample+flatten)
    print("\nLoading MRIs, resampling to common grid, flattening...")
    X = load_dataset_matrix(matched_paths, voxel_size_mm=args.voxel_size_mm)
    print(f"Feature matrix X shape: {X.shape}  (subjects, voxels)")
    print(f"Labels y shape: {y.shape}")

    # 6) Cross-validation (leakage-free pipeline)
    print("\nRunning cross-validation...")
    run_cross_validation(
        X=X,
        y=y,
        n_splits=args.n_splits,
        shuffle=True if args.shuffle else True,  # default to True for safer splits
        random_state=args.random_state,
        n_components=args.n_components,
        n_estimators=args.n_estimators,
        stratified=args.stratified,
    )


if __name__ == "__main__":
    main()
