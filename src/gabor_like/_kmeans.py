import torch
from torch.nn.functional import normalize


def _spherical_kmeans(k: int, vectors: torch.Tensor, max_iters: int) -> torch.Tensor:
    vector_amount = vectors.size(dim=0)
    centroid_indexes = torch.randperm(vector_amount)[:k]
    centroids = vectors[centroid_indexes]

    vectors = normalize(vectors, dim=1)
    centroids = normalize(centroids, dim=1)

    iter = 0
    while iter <= max_iters:
        cos_sim = vectors @ centroids.T
        labels = torch.argmax(cos_sim, dim=1)

        centroid_list = []
        for i in range(k):
            vec_in_cluster = vectors[labels == i]

            if vec_in_cluster.size() == 0:
                centroid = centroids[i]
            else:
                centroid = torch.mean(vec_in_cluster, dim=0)
            centroid_list.append(centroid)
        centroids = torch.stack(centroid_list)
        centroids = normalize(centroids, dim=1)

        iter += 1

    cos_sim = vectors @ centroids.T
    labels = torch.argmax(cos_sim, dim=1)
    return labels
