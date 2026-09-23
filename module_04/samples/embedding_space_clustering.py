"""Embedding space and clustering.

Documents about the same topic land near each other in embedding space.
KMeans clustering on the raw embeddings should recover the 4 topics in
corpus.py without ever being told what the topics are.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from llm_client import embed

from corpus import CATEGORIES, CORPUS


def main() -> None:
    embeddings = np.array(embed(CORPUS))

    n_clusters = len(set(CATEGORIES))  # 4 known topics
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    cluster_ids = kmeans.fit_predict(embeddings)

    print(f"Clustering {len(CORPUS)} documents into {n_clusters} groups using only embeddings:\n")
    for cluster_id in sorted(set(cluster_ids)):
        print(f"Cluster {cluster_id}:")
        for doc, cid, true_category in zip(CORPUS, cluster_ids, CATEGORIES):
            if cid == cluster_id:
                print(f"  [{true_category:8s}] {doc}")
        print()

    print("Compare the bracketed ground-truth topic within each cluster -- ")
    print("documents from the same true topic should mostly land in the same cluster.")


if __name__ == "__main__":
    main()
