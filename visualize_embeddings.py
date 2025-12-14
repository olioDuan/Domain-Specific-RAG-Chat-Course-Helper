# visualize_embeddings.py

from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import config
from loader import load_all_slides
from embedder import get_embedding_model


def compute_embeddings():
    """
    Load all slides (one page per Document) and compute embeddings.
    Returns:
        embeddings: np.ndarray, shape (N, D)
        lectures:   list[str], length N, lecture id for each point (e.g., 'Lec 1')
        pages:      list[int], length N, page index for each point
    """
    print(f"[PCA] Loading slides from: {config.SLIDES_DIR}")
    docs = load_all_slides(config.SLIDES_DIR)
    print(f"[PCA] Loaded {len(docs)} documents (slide-level chunks).")

    texts = [doc.page_content for doc in docs]
    lectures = [doc.metadata.get("lecture", "Unknown") for doc in docs]
    pages = [doc.metadata.get("page", -1) for doc in docs]

    print("[PCA] Initializing embedding model...")
    embed_model = get_embedding_model()

    print("[PCA] Computing embeddings...")
    emb_list = embed_model.embed_documents(texts)
    embeddings = np.array(emb_list, dtype=np.float32)

    print(f"[PCA] Embeddings shape: {embeddings.shape}")
    return embeddings, lectures, pages


def run_pca(embeddings: np.ndarray, n_components: int = 2):
    """
    Run PCA on embeddings for dimensionality reduction.
    """
    print(f"[PCA] Running PCA to {n_components} dimensions...")
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    print(f"[PCA] Explained variance ratio: {explained}")
    return reduced, explained


def plot_pca(reduced: np.ndarray, lectures: list[str]):
    """
    Plot a 2D PCA scatter colored by lecture and save it.
    """
    data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "lecture_pca.png"

    print("[PCA] Plotting scatter plot...")

# 1. Prepare mapping
    unique_lecs = sorted(list(set(lectures)), key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 999)
    
    lecture_to_idx = {lec: i for i, lec in enumerate(unique_lecs)}
    color_ids = np.array([lecture_to_idx[lec] for lec in lectures], dtype=int)

    plt.figure(figsize=(10, 8)) # Slightly larger so the legend is not crowded

    # 2. Get colormap 
    # n_colors set to number of lectures
    import matplotlib.cm as cm
    num_lecs = len(unique_lecs)
    
    cmap_name = 'jet'
    cmap = plt.get_cmap(cmap_name, num_lecs)

    # 3. Plot scatter
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=color_ids,
        cmap=cmap, 
        alpha=0.7,
        s=30,
        vmin=0, 
        vmax=num_lecs-1 
    )

    # 4. Build legend
    handles = []
    for i, lec in enumerate(unique_lecs):
        color = cmap(i) 
        
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                color=color, 
                label=lec,
            )
        )

    plt.legend(
        handles=handles,
        title="Lecture",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Lecture Slide Embeddings")
    plt.grid(True, linestyle='--', alpha=0.3) 

    plt.tight_layout()
    plt.savefig(out_path, dpi=300) 
    plt.close()

    print(f"[PCA] Saved PCA plot to: {out_path}")


def plot_lecture_centers(reduced: np.ndarray, lectures: list[str]):
    """
    Optional: plot lecture centers by averaging points per lecture.
    """
    data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "lecture_centers_pca.png"

    # Group all points by lecture
    clusters = defaultdict(list)
    for coord, lec in zip(reduced, lectures):
        clusters[lec].append(coord)

    centers = []
    labels = []
    for lec, coords in clusters.items():
        arr = np.vstack(coords)
        centers.append(arr.mean(axis=0))
        labels.append(lec)

    centers = np.vstack(centers)

    plt.figure(figsize=(8, 6))
    plt.scatter(centers[:, 0], centers[:, 1])

    for (x, y), lec in zip(centers, labels):
        plt.text(x, y, lec, fontsize=8, ha="center", va="center")

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Lecture Centers")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[PCA] Saved lecture centers plot to: {out_path}")


def main():
    embeddings, lectures, pages = compute_embeddings()
    reduced, explained = run_pca(embeddings, n_components=2)
    plot_pca(reduced, lectures)
    plot_lecture_centers(reduced, lectures)
    print("[PCA] Done.")


if __name__ == "__main__":
    main()