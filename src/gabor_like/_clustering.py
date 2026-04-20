import torch
from ._tensor_ops import _normalize_in_place


def _spherical_kmeans(k: int, vectors: torch.Tensor, max_iters: int) -> torch.Tensor:
    """
    Run spherical k-means and return cluster label per vector.

    Args:
        k: Number of clusters.
        vectors: Input tensor with shape (n_vectors, n_features).
            This tensor is normalized in place.
        max_iters: Maximum number of iterations.

    Returns:
        Tensor of shape (n_vectors) with integer cluster ids in [0, k-1].

    Notes:
        - Initialization uses random samples from input vectors.
        - Random seed fixed to 42 for deterministic centroid init.
        - Empty cluster keeps previous centroid.
    """
    vector_amount = vectors.size(dim=0)
    torch.manual_seed(42)
    centroid_indexes = torch.randperm(vector_amount)[:k]
    centroids = vectors[centroid_indexes]

    _normalize_in_place(vectors, dim=1)
    _normalize_in_place(centroids, dim=1)

    iter_idx = 0
    while iter_idx < max_iters:
        cos_sim = vectors @ centroids.T
        labels = torch.argmax(cos_sim, dim=1)

        centroid_list = []
        for i in range(k):
            vec_in_cluster = vectors[labels == i]

            if vec_in_cluster.size(dim=0) == 0:
                centroid = centroids[i]
            else:
                centroid = torch.mean(vec_in_cluster, dim=0)
            centroid_list.append(centroid)
        centroids = torch.stack(centroid_list)
        _normalize_in_place(centroids, dim=1)

        iter_idx += 1
        del cos_sim

    cos_sim = vectors @ centroids.T
    labels = torch.argmax(cos_sim, dim=1)
    return labels
