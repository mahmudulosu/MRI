import os
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


VALID_AGES = ["20-25", "25-30", "60-65", "65-70", "70-75"]
AGE_GROUP_MAPPING = {
    "20-25": "young",
    "25-30": "young",
    "60-65": "old",
    "65-70": "old",
    "70-75": "old",
}


def extract_participant_id(mri_filename: str) -> str:
    # Extracts "sub-010002" from "sub-010002_ses-01_acq-mp2rage_brain.nii.gz"
    return mri_filename.split("_")[0]


def load_mri(file_path: str, target_shape=(128, 128, 128)) -> np.ndarray:
    """
    Load a NIfTI, normalize to [0,1], and resize to target_shape using scipy zoom.
    Returns float32 array of shape target_shape.
    """
    img = nib.load(file_path).get_fdata()

    # Normalize safely
    img = img.astype(np.float32)
    vmin, vmax = np.min(img), np.max(img)
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    # Resize via zoom
    zoom_factors = (
        target_shape[0] / img.shape[0],
        target_shape[1] / img.shape[1],
        target_shape[2] / img.shape[2],
    )
    img_resized = zoom(img, zoom_factors, order=1)  # linear interpolation
    img_resized = img_resized.astype(np.float32)

    return img_resized


def build_model(input_shape):
    model = Sequential([
        Conv3D(16, kernel_size=(3, 3, 3), activation="relu", input_shape=input_shape),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Conv3D(32, kernel_size=(3, 3, 3), activation="relu"),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Conv3D(64, kernel_size=(3, 3, 3), activation="relu"),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="3D CNN age-group classifier (young vs old) from NIfTI volumes.")
    parser.add_argument("--mri_data_path", type=str, required=True, help="Folder containing .nii/.nii.gz MRI files")
    parser.add_argument("--age_csv_path", type=str, required=True, help="Participants CSV with participant_id and age")
    parser.add_argument("--target_shape", nargs=3, type=int, default=[128, 128, 128], help="Resize target (x y z)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--limit", type=int, default=0, help="Optional: limit number of samples for quick testing")
    args = parser.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)

    # Reproducibility
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)

    # Load age data
    age_data = pd.read_csv(args.age_csv_path)
    if "participant_id" not in age_data.columns or "age" not in age_data.columns:
        raise ValueError(f"CSV must contain participant_id and age. Found: {list(age_data.columns)}")

    age_data = age_data[age_data["age"].isin(VALID_AGES)].copy()
    age_data["age_group"] = age_data["age"].map(AGE_GROUP_MAPPING)

    # Fix potential "sub-sub-" issue (keep your original fix)
    age_data["participant_id"] = age_data["participant_id"].astype(str).apply(lambda x: x.replace("sub-sub-", "sub-"))

    # Encode labels
    le = LabelEncoder()
    age_data["label"] = le.fit_transform(age_data["age_group"])  # young=0, old=1 (usually)
    label_map = {cls: int(i) for i, cls in enumerate(le.classes_)}

    # List MRI files
    mri_files = [
        f for f in os.listdir(args.mri_data_path)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    print(f"📂 Found {len(mri_files)} MRI files. Example: {mri_files[:5]}")

    age_ids = set(age_data["participant_id"].values.tolist())

    # Filter MRI files that match age data
    filtered_mri_files = [f for f in mri_files if extract_participant_id(f) in age_ids]
    print(f"✅ Matched {len(filtered_mri_files)} MRI files with age data.")

    if args.limit and args.limit > 0:
        filtered_mri_files = filtered_mri_files[:args.limit]
        print(f"🔧 Limiting to first {len(filtered_mri_files)} samples for quick test.")

    if len(filtered_mri_files) == 0:
        raise SystemExit("❌ No valid MRI files matched with age data. Check filenames and participant IDs.")

    # Load MRI data
    X, y = [], []
    target_shape = tuple(args.target_shape)

    for mri_file in tqdm(filtered_mri_files, desc="Loading MRI data"):
        pid = extract_participant_id(mri_file)
        file_path = os.path.join(args.mri_data_path, mri_file)

        try:
            img_arr = load_mri(file_path, target_shape=target_shape)
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            continue

        X.append(img_arr)

        label_row = age_data.loc[age_data["participant_id"] == pid, "label"]
        if label_row.empty:
            continue
        y.append(int(label_row.values[0]))

    if len(X) == 0:
        raise SystemExit("❌ No MRI volumes loaded successfully. Check file integrity and preprocessing.")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, X, Y, Z, 1)
    y = to_categorical(np.array(y, dtype=np.int32), num_classes=2)

    print(f"Loaded volumes: X={X.shape}, y={y.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=np.argmax(y, axis=1)
    )

    # Build + train model
    model = build_model(input_shape=X_train.shape[1:])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_acc:.3f}")

    # Save outputs
    model_path = os.path.join(args.outputs_dir, "mri_3dcnn_age_classifier.h5")
    model.save(model_path)

    mapping_path = os.path.join(args.outputs_dir, "label_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"classes": list(le.classes_), "mapping": label_map}, f, indent=2)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_path = os.path.join(args.outputs_dir, "training_history.csv")
    hist_df.to_csv(hist_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved label mapping: {mapping_path}")
    print(f"Saved training history: {hist_path}")


if __name__ == "__main__":
    main()
