import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

VALID_AGES = ["20-25", "25-30", "60-65", "65-70", "70-75"]
AGE_GROUP_MAPPING = {
    "20-25": "young",
    "25-30": "young",
    "60-65": "old",
    "65-70": "old",
    "70-75": "old",
}


def load_and_merge(mri_feature_path: str, age_csv_path: str) -> pd.DataFrame:
    mri_features = pd.read_csv(mri_feature_path)
    age_data = pd.read_csv(age_csv_path)

    age_data = age_data[age_data["age"].isin(VALID_AGES)].copy()
    age_data["age_group"] = age_data["age"].map(AGE_GROUP_MAPPING)

    merged = pd.merge(mri_features, age_data, on="participant_id", how="inner")
    return merged


def plot_feature(merged_data: pd.DataFrame, feature: str, outputs_dir: str):
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=merged_data,
        x=feature,
        hue="age_group",
        kde=True,
        bins=30,
        palette="coolwarm",
    )
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(f"Distribution of {feature} by Age Group")
    plt.legend(title="Age Group")
    plt.tight_layout()

    out_path = os.path.join(outputs_dir, f"{feature}_distribution.png")
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot WM/GM/CSF voxel-count distributions by age group.")
    parser.add_argument("--mri_feature_path", type=str, required=True)
    parser.add_argument("--age_csv_path", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)
    sns.set_style("whitegrid")

    merged = load_and_merge(args.mri_feature_path, args.age_csv_path)
    if merged.empty:
        raise SystemExit("❌ No matching MRI and age data found. Ensure participant IDs match.")

    for feat in ["WM_voxels", "GM_voxels", "CSF_voxels"]:
        if feat not in merged.columns:
            print(f"Skipping missing column: {feat}")
            continue
        plot_feature(merged, feat, args.outputs_dir)


if __name__ == "__main__":
    main()
