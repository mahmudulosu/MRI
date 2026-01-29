import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn.image import resample_img
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_and_flatten_mri(file_path: str) -> np.ndarray:
    img = nib.load(file_path)
    target_affine = np.eye(3) * 2.0  # 2mm isotropic
    img_resampled = resample_img(img, target_affine=target_affine, interpolation="continuous")
    return img_resampled.get_fdata().astype(np.float32).ravel()


def apply_pca_to_dataset(dataset_folder: str, n_components: int = 50):
    file_paths = [
        os.path.join(dataset_folder, f)
        for f in os.listdir(dataset_folder)
        if os.path.isfile(os.path.join(dataset_folder, f)) and (f.endswith(".nii") or f.endswith(".nii.gz"))
    ]
    file_paths.sort()

    all_data = [load_and_flatten_mri(fp) for fp in file_paths]
    X = np.stack(all_data, axis=0)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print("PCA Components shape:", X_pca.shape)
    return X_pca, pca, file_paths


def plot_pca_components(X_pca: np.ndarray, participant_index: int, file_paths):
    plt.figure(figsize=(10, 6))
    plt.plot(X_pca[participant_index], marker="o")
    plt.title(f"PCA Components for {os.path.basename(file_paths[participant_index])}")
    plt.xlabel("Component Number")
    plt.ylabel("PCA Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], s=20)
        plt.scatter(X_pca[participant_index, 0], X_pca[participant_index, 1], color="red", s=60)
        plt.title("Scatter plot of the first two PCA components")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    csv_path = r"C:\research\MRI\participants_LSD_andLEMON.csv"
    data_folder = r"C:\research\MRI\structural_MRI"

    age_data = pd.read_csv(csv_path)
    print("Columns in CSV:", list(age_data.columns))

    valid_ages = ["20-25", "25-30", "60-65", "65-70", "70-75"]
    age_group_mapping = {"20-25": "young", "25-30": "young",
                         "60-65": "old", "65-70": "old", "70-75": "old"}

    filtered_data = age_data[age_data["age"].isin(valid_ages)].copy()
    age_mapping = {row["participant_id"]: age_group_mapping[row["age"]] for _, row in filtered_data.iterrows()}

    # PCA over all files
    pca_components, pca, file_paths = apply_pca_to_dataset(data_folder, n_components=50)

    # Match labels
    labels, matched_paths, X_valid = [], [], []
    for i, fp in enumerate(file_paths):
        subject_id = os.path.basename(fp).split("_")[0]
        if subject_id in age_mapping:
            labels.append(age_mapping[subject_id])
            matched_paths.append(fp)
            X_valid.append(pca_components[i])

    labels = np.array(labels)
    X_valid = np.array(X_valid)

    if len(X_valid) == 0:
        raise SystemExit("No MRI files matched participant IDs in the CSV.")

    label_mapping = {"young": 0, "old": 1}
    y = np.array([label_mapping[l] for l in labels], dtype=int)

    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(X_valid, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["young", "old"]))

    # Example plot
    plot_pca_components(X_valid, 0, matched_paths)


if __name__ == "__main__":
    main()
