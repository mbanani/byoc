"""
Code to find correspondances between two point clouds
"""
import torch
from pytorch3d.ops.knn import knn_gather, knn_points
from torch.nn.functional import cosine_similarity, log_softmax, normalize

from ..utils.util import nn_gather


@torch.jit.script
def calculate_ratio_test(dists):
    """
    Calculate weights for matches based on the ratio between kNN distances.

    Input:
        (N, P, 2) Cosine Distance between point and nearest 2 neighbors
    Output:
        (N, P, 1) Weight based on ratio; higher is more unique match
    """
    # Convert points so that 0 means perfect similarity and clamp to avoid numerical
    # instability
    dists = (1 - dists).clamp(min=1e-9)

    # Ratio -- close to 0 is completely unique; 1 is same feature
    ratio = dists[:, :, 0:1] / dists[:, :, 1:2]
    # Weight -- Convert so that higher is more unique
    weight = 1 - ratio

    return weight


@torch.jit.script
def get_topk_matches(dists, idx, num_corres: int):
    num_corres = min(num_corres, dists.shape[1])
    dist, idx_source = torch.topk(dists, k=num_corres, dim=1)
    idx_target = idx.gather(1, idx_source)
    return idx_source, idx_target, dist


def get_correspondences(P1, P2, num_corres, metric="cosine", ratio_test=True):
    """
    Finds the kNN according to either euclidean distance or cosine distance. This is
    tricky since PyTorch3D's fast kNN kernel does euclidean distance, however, we can
    take advantage of the relation between euclidean distance and cosine distance for
    points sampled on an n-dimension sphere.

    Using the quadratic expansion, we find that finding the kNN between two normalized
    is the same regardless of whether the metric is euclidean distance or cosine
    similiarity.

        -2 * xTy = (x - y)^2 - x^2 - y^2
        -2 * xtY = (x - y)^2 - 1 - 1
        - xTy = 0.5 * (x - y)^2 - 1

    Hence, the metric that would maximize cosine similarity is the same as that which
    would minimize the euclidean distance between the points, with the distances being
    a simple linear transformation.

    Input:
        P1          pytorch3d's Pointclouds i   features for first pointcloud
        P2          pytorch3d's Pointclouds i   features for first pointcloud
        num_corres  Int                         number of correspondances
        metric      {cosine, euclidean}         metric to be used for kNN
        ratio_test  Boolean                     whether to use ratio test for kNN

    Returns:
        LongTensor (N x 2 * num_corres)         Indices for first pointcloud
        LongTensor (N x 2 * num_corres)         Indices for second pointcloud
        FloatTensor (N x 2 * num_corres)        Weights for each correspondace
        FloatTensor (N x 2 * num_corres)        Cosine distance between features
    """
    P1_F = P1.features_padded()
    P2_F = P2.features_padded()
    P1_N = P1.num_points_per_cloud()
    P2_N = P2.num_points_per_cloud()

    # Normalize features if needed
    assert metric in ["euclidean", "cosine"]
    if metric == "cosine":
        P1_F = torch.nn.functional.normalize(P1_F, dim=2)
        P2_F = torch.nn.functional.normalize(P2_F, dim=2)

    # Calculate kNN for k=2; both outputs are (N, P, K)
    # idx_1 returns the indices of the nearest neighbor in P2
    K = 2 if ratio_test else 1
    dists_1, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=K)
    dists_2, idx_2, _ = knn_points(P2_F, P1_F, P2_N, P1_N, K=K)

    # Take the nearest neighbor for the indices for k={1, 2}
    idx_1 = idx_1[:, :, 0:1]
    idx_2 = idx_2[:, :, 0:1]

    # Transform euclidean distance of points on a sphere to cosine similarity
    cosine_1 = 1 - 0.5 * dists_1
    cosine_2 = 1 - 0.5 * dists_2

    if metric == "cosine":
        dists_1 = cosine_1
        dists_2 = cosine_2

    # Apply ratio test
    if ratio_test:
        weights_1 = calculate_ratio_test(dists_1)
        weights_2 = calculate_ratio_test(dists_2)
    else:
        weights_1 = dists_1[:, :, 0:1]
        weights_2 = dists_2[:, :, 0:1]

    # Get topK matches in both directions
    n_corres_1 = min(num_corres, P1_N.min())
    n_corres_2 = min(num_corres, P2_N.min())
    if n_corres_1 < num_corres or n_corres_2 < num_corres:
        print(f"Min corresponds is {n_corres_1} and {n_corres_2}")

    m12_idx1, m12_idx2, m12_dist = get_topk_matches(weights_1, idx_1, n_corres_1)
    m21_idx2, m21_idx1, m21_dist = get_topk_matches(weights_2, idx_2, n_corres_2)
    cosine_1 = cosine_1[:, :, 0:1].gather(1, m12_idx1)
    cosine_2 = cosine_2[:, :, 0:1].gather(1, m21_idx2)

    # concatenate into correspondances and weights
    matches_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1).squeeze(dim=2)
    matches_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1).squeeze(dim=2)
    matches_dist = torch.cat((m12_dist, m21_dist), dim=1).squeeze(dim=2)
    matches_cosn = torch.cat((cosine_1, cosine_2), dim=1).squeeze(dim=2)

    return matches_idx1, matches_idx2, matches_dist, matches_cosn


def transfer_correspondances(corr, pc_0, pc_1, v2g_cfg, project_fn, residual):
    # get match features and coord
    corr_F_0 = nn_gather(pc_0.features_padded(), corr[0])
    corr_F_1 = nn_gather(pc_1.features_padded(), corr[1])

    # get weights for visualization
    v2g_weight = cosine_similarity(corr_F_0, corr_F_1, dim=2)
    v2g_weight = (1 + v2g_weight) / 2.0

    # project features -- corr_F_{0, 1} are B x N x F -- poiloud features
    assert project_fn is not None
    corr_F_0_p = project_fn(corr_F_0)
    corr_F_1_p = project_fn(corr_F_1)

    # calculate loss - sim_s is [0, 4]
    sim_0 = cosine_similarity(corr_F_0.detach(), corr_F_1_p, dim=2)
    sim_1 = cosine_similarity(corr_F_0_p, corr_F_1.detach(), dim=2)
    sim_s = 2 + sim_0 + sim_1

    """
    Three weighting schemes:
        - residual: incorporate residual alignment error
        - lowe: use the lowe weights from visual correpsonda
        - none: classic sim siam
    """
    if v2g_cfg.weight == "residual":
        sim_s = normalize(sim_s, p=1, dim=1)
        v2g_loss = (sim_s * residual.detach()).sum(dim=1)
    elif v2g_cfg.weight == "lowe":
        lowe_dist = normalize(corr[2].detach(), p=1, dim=1)
        v2g_loss = ((4 - sim_s) * lowe_dist).sum(dim=1)
    elif v2g_cfg.weight == "none":
        v2g_loss = 4 - sim_s.mean(dim=1)
    else:
        raise ValueError()

    return v2g_loss, v2g_weight
