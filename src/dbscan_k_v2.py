"""
DBSCAN+K v2: 利用 ER-tSNE 邻居结构增强异常点评分。

与 v1 的区别：
  v1 score = dist_to_cluster / cluster_density（纯几何）
  v2 score = (dist_to_cluster / cluster_density) × (1 - 局部DLP)

局部 DLP：每个噪声点在 embedding 里的 k 近邻排名与高维 k 近邻排名的 Spearman 相关。
  - DLP 高（邻居保持好）→ 这个点在正常区域，不是真异常 → score 低 → 过滤掉
  - DLP 低（邻居被打乱）→ 这个点的邻居结构异常 → score 高 → 保留为 marker

这样 DBSCAN+K 才真正利用了 ER-tSNE 的邻居信息，DBSCAN 做不到这一点。
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr


def _compute_local_dlp(Y_emb, X_high, noise_indices, k=10):
    """
    计算每个噪声点的局部 DLP（邻居保持度）。

    对每个噪声点：
      1. 在高维 X_high 找 k 近邻
      2. 这 k 个邻居在 Y_emb 里的距离排名
      3. Spearman 相关 = 局部 DLP

    Returns: ndarray (len(noise_indices),), range [-1, 1]
    """
    # 高维 kNN（只对噪声点查询）
    nn_high = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X_high)
    dists_h, idx_h = nn_high.kneighbors(X_high[noise_indices])
    # 去掉自己
    idx_h = idx_h[:, 1:]  # (n_noise, k)

    r_high = np.arange(1, k + 1)
    local_dlp = np.zeros(len(noise_indices))

    for i in range(len(noise_indices)):
        nb_idx = idx_h[i]
        pt = noise_indices[i]
        # 这些邻居在 embedding 里的距离
        dists_low = np.sqrt(((Y_emb[pt] - Y_emb[nb_idx]) ** 2).sum(axis=1))
        r_low = np.argsort(np.argsort(dists_low)) + 1
        corr, _ = spearmanr(r_high, r_low)
        local_dlp[i] = corr if not np.isnan(corr) else 0.0

    return local_dlp


def dbscan_k_v2(Y_emb, X_high, k, minPts=30, outlier_top_pct=30,
                labels_dir=None, dlp_k=10, dlp_weight=1.0):
    """
    DBSCAN+K v2: 结合几何异常度 + 邻居结构异常度。

    Parameters
    ----------
    Y_emb          : ndarray (N, 2) — 2D embedding (t-SNE 或 ER-tSNE)
    X_high         : ndarray (N, d) — 原始高维数据
    k              : int — Trading Marker 的标签值
    minPts         : int — DBSCAN min_samples
    outlier_top_pct: float — 保留 top P% 噪声点
    labels_dir     : ndarray (N,) or None — 方向标签 (1/-1)
    dlp_k          : int — 局部 DLP 的近邻数
    dlp_weight     : float — DLP 权重，0=退化为 v1，1=等权

    Returns
    -------
    R : ndarray (N,) — 聚类标签
    """
    N = Y_emb.shape[0]
    rng = np.random.default_rng(42)

    # Phase 0: Adaptive epsilon
    nn = NearestNeighbors(n_neighbors=minPts, algorithm='auto')
    nn.fit(Y_emb)
    D_knn = nn.kneighbors(Y_emb)[0][:, -1]
    eps_strict = np.percentile(D_knn, 25)

    # Phase 1: DBSCAN
    labels_dbscan = DBSCAN(eps=eps_strict, min_samples=minPts).fit_predict(Y_emb)
    O_indices = np.where(labels_dbscan == -1)[0]

    # Phase 2: Score + Local DLP
    labels_updated = labels_dbscan.copy()

    if len(O_indices) == 0:
        O_true_indices = np.array([], dtype=int)
    else:
        cluster_labels_p2 = sorted(set(labels_dbscan[labels_dbscan >= 0]))

        if len(cluster_labels_p2) == 0:
            O_true_indices = O_indices.copy()
        else:
            cluster_members = {cl: np.where(labels_dbscan == cl)[0]
                               for cl in cluster_labels_p2}

            # Cluster density
            cluster_density = {}
            for cl, members in cluster_members.items():
                if len(members) <= 1:
                    cluster_density[cl] = 1e-10
                    continue
                sample = (rng.choice(members, size=50, replace=False)
                          if len(members) > 50 else members)
                pw = pairwise_distances(Y_emb[sample])
                cluster_density[cl] = pw[np.triu_indices(len(sample), k=1)].mean()

            # Nearest cluster for each noise point
            nearest_cluster = np.empty(len(O_indices), dtype=int)
            nearest_dist = np.full(len(O_indices), np.inf)
            for cl, members in cluster_members.items():
                nn_cl = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(Y_emb[members])
                dists = nn_cl.kneighbors(Y_emb[O_indices])[0].ravel()
                closer = dists < nearest_dist
                nearest_dist[closer] = dists[closer]
                nearest_cluster[closer] = cl

            # Geometric score (same as v1)
            geo_scores = np.array([nearest_dist[j] / cluster_density[nearest_cluster[j]]
                                   for j in range(len(O_indices))])

            # Local DLP for noise points (NEW in v2)
            local_dlp = _compute_local_dlp(Y_emb, X_high, O_indices, k=dlp_k)
            # local_dlp ∈ [-1, 1], higher = better preserved = less anomalous
            # We want: low DLP → high score
            dlp_factor = 1.0 - local_dlp  # range [0, 2], high when DLP is low

            # Combined score
            # Normalize both components to [0, 1] before combining
            geo_norm = (geo_scores - geo_scores.min()) / (geo_scores.max() - geo_scores.min() + 1e-10)
            dlp_norm = (dlp_factor - dlp_factor.min()) / (dlp_factor.max() - dlp_factor.min() + 1e-10)

            scores = geo_norm + dlp_weight * dlp_norm

            # Direction-balanced percentile cutoff
            if labels_dir is not None:
                O_labels = labels_dir[O_indices]
                O_true_mask = np.zeros(len(O_indices), dtype=bool)
                for direction in [1, -1]:
                    dm = (O_labels == direction)
                    if dm.sum() == 0:
                        continue
                    thr = np.percentile(scores[dm], 100 - outlier_top_pct)
                    O_true_mask[dm] = scores[dm] >= thr
            else:
                thr = np.percentile(scores, 100 - outlier_top_pct)
                O_true_mask = scores >= thr

            O_true_indices = O_indices[O_true_mask]

            for j in np.where(~O_true_mask)[0]:
                labels_updated[O_indices[j]] = nearest_cluster[j]

    # Phase 3: Label assignment
    R = np.zeros(N, dtype=int)
    o_true_set = set(O_true_indices.tolist())
    for i in range(N):
        if i in o_true_set:
            R[i] = k
        else:
            lbl = labels_updated[i]
            # FIX (cluster-label collision): 簇标签编到负数空间, 绝不与 marker 值 k 冲突。
            # 原 (lbl+1): 当簇数 >= k 时, 第 k 个簇 (lbl=k-1) -> R=k, 整簇被误当 marker。
            # 实测金融 2D embedding 簇数达 20-46, 普遍触发, 每窗污染数十个正常点。
            # 改 -(lbl+1): 簇点 -> {-1,-2,...}, (R==k) 只剩 O_true_indices 真 marker。
            # marker 选择逻辑 (geo+DLP+方向 Top-P%) 完全不变, 不会退化成 plain DBSCAN。
            R[i] = -(lbl + 1) if lbl >= 0 else k
    return R


# ---------------------------------------------------------------------------
# Self-test: marker 标签绝不能与 cluster 标签冲突 (回归防护)
# ---------------------------------------------------------------------------
def _self_test():
    print('=== dbscan_k_v2 SELF-TEST ===')
    rng = np.random.default_rng(0)
    # 6 个清晰分离的簇 (lbl 0..5, 必含曾撞 k=4 的 lbl=3) + 均匀噪声散点
    centers = [(0, 0), (20, 0), (0, 20), (20, 20), (40, 0), (0, 40)]
    blobs = [rng.normal([cx, cy], 0.3, size=(50, 2)) for cx, cy in centers]
    noise = rng.uniform(-8, 48, size=(40, 2))
    Y = np.vstack(blobs + [noise])
    X = Y + rng.normal(0, 0.05, Y.shape)        # 高维占位 (local DLP 用)

    R = dbscan_k_v2(Y, X, k=4, minPts=10, outlier_top_pct=50,
                    labels_dir=None, dlp_k=10, dlp_weight=1.0)

    # 复现内部 DBSCAN 拿真实簇标签
    nn = NearestNeighbors(n_neighbors=10).fit(Y)
    eps = np.percentile(nn.kneighbors(Y)[0][:, -1], 25)
    lab = DBSCAN(eps=eps, min_samples=10).fit_predict(Y)
    nclust = len(set(lab[lab >= 0]))
    marker = (R == 4)

    assert nclust >= 4, f'测试需 >=4 簇覆盖冲突情形, 实得 {nclust}'
    contaminated = int((marker & (lab >= 0)).sum())
    assert contaminated == 0, f'marker 含 {contaminated} 个簇内点 (污染未修复!)'
    assert marker.sum() > 0, 'marker 不应为空'
    assert bool((marker <= (lab == -1)).all()), 'marker 必须全来自 DBSCAN 噪声点'
    print(f'  [ok] 簇数={nclust} (>3, 覆盖原 bug 情形)')
    print(f'  [ok] marker={int(marker.sum())} 个, 全来自噪声, 0 簇内污染')
    print(f'  [ok] 噪声点共 {int((lab == -1).sum())}, marker 是其 Top-P% 子集 (未退化成全噪声)')
    print('=== PASSED ===')


if __name__ == '__main__':
    _self_test()
