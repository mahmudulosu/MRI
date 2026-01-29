import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main(valid_pca_components, y, matched_file_paths):
    # First 8 unique subject IDs
    unique_ids = np.unique([os.path.basename(fp).split("_")[0] for fp in matched_file_paths])
    first_eight = unique_ids[:8]

    mask = np.array([os.path.basename(fp).split("_")[0] in first_eight for fp in matched_file_paths])

    X_8 = valid_pca_components[mask]
    ids_8 = np.array([os.path.basename(fp).split("_")[0] for fp in np.array(matched_file_paths)[mask]])
    y_8 = y[mask]

    # Scale then t-SNE
    X_8_scaled = StandardScaler().fit_transform(X_8)

    perplexity = max(2, min(5, len(X_8_scaled) - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_8_scaled)

    # t-SNE plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        hue=y_8,
        style=ids_8,
        palette="Set1",
        s=80
    )
    plt.title("t-SNE of PCA Features (first 8 participants)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # PCA feature curves
    for sid in first_eight:
        idx = np.where(ids_8 == sid)[0]
        if len(idx) == 0:
            continue
        plt.figure(figsize=(12, 5))
        plt.plot(X_8[idx[0]], marker="o")
        plt.title(f"PCA components for {sid}")
        plt.xlabel("Component #")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print(
        "This script expects valid_pca_components, y, matched_file_paths from snippet (01).\n"
        "In practice, you should import them or save/load arrays.\n"
        "Tip: run snippet (01) first, then copy arrays here, or save as .npz."
    )
