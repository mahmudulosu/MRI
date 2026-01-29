import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.image import resample_img
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_and_flatten_mri(file_path: str) -> np.ndarray:
    img = nib.load(file_path)
    target_affine = np.eye(3) * 2.0
    img_resampled = resample_img(img, target_affine=target_affine, interpolation="continuous")
    return img_resampled.get_fdata().astype(np.float32).ravel()


def apply_pca_to_dataset(dataset_folder: str, n_components: int):
    file_paths = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
                  if (f.endswith(".nii") or f.endswith(".nii.gz"))]
    file_paths.sort()
    X = np.stack([load_and_flatten_mri(fp) for fp in file_paths], axis=0)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, file_paths


def main():
    parser = argparse.ArgumentParser(description="Gender classification (PCA voxel features + RF, 10-fold CV).")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = pd.read_csv(args.csv_path)
    meta["gender"] = meta["gender"].astype(str).str.upper().str.strip()
    meta["participant_id"] = meta["participant_id"].astype(str)

    label_map = {"F": 0, "M": 1}
    pid_to_y = {r["participant_id"]: label_map[r["gender"]]
                for _, r in meta.iterrows() if r["gender"] in label_map}

    X_pca, paths = apply_pca_to_dataset(args.data_folder, args.n_components)

    X_valid, y = [], []
    for i, fp in enumerate(paths):
        pid = os.path.basename(fp).split("_")[0]
        if pid in pid_to_y:
            X_valid.append(X_pca[i])
            y.append(pid_to_y[pid])

    X_valid = np.asarray(X_valid, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    if len(X_valid) == 0:
        raise SystemExit("No MRI files matched CSV participant IDs.")

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    accs = []

    for fold, (tr, te) in enumerate(kf.split(X_valid), 1):
        Xtr, Xte = X_valid[tr], X_valid[te]
        ytr, yte = y[tr], y[te]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        clf = RandomForestClassifier(n_estimators=200, random_state=args.seed)
        clf.fit(Xtr_s, ytr)

        pred = clf.predict(Xte_s)
        acc = accuracy_score(yte, pred)
        accs.append(acc)
        print(f"Fold {fold:02d} Accuracy: {acc*100:.2f}%")

    print(f"\nMean Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")


if __name__ == "__main__":
    main()
