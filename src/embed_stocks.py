"""Computes t-SNE (Barnes-Hut) and SR-tSNE (lambda=1.0) for every CSV in data/ and writes them to cache/."""
import sys, os, time, glob
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.sparse import coo_matrix
from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sr_tsne import run_sr_tsne

DATA = 'data'
CACHE = 'cache'
K_LOSS, LAM = 10, 1.0


def load_feats(tag):
    df = pd.read_csv(f'{DATA}/{tag}_n.csv')
    cols = [c for c in df.columns if c != 'Date']       # 8 feat: OHLCV+avg+notional+cot
    return StandardScaler().fit_transform(df[cols].values.astype(np.float64))


def embed(tag):
    X = load_feats(tag)
    n = X.shape[0]
    Ytsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000,
                 method='barnes_hut', learning_rate='auto', init='pca').fit_transform(X)
    np.save(f'{CACHE}/{tag}_tsne.npy', Ytsne)
    Dsq = pairwise_distances(X, metric='sqeuclidean')
    Pc = _joint_probabilities(Dsq, 30, 0)
    Pd = squareform(Pc); np.fill_diagonal(Pd, 0.0)
    Pcoo = coo_matrix(Pd); del Pd, Dsq
    nb = min(n - 1, 90)
    nidx = NearestNeighbors(n_neighbors=nb + 1, n_jobs=-1).fit(X).kneighbors(
        return_distance=False)[:, 1:K_LOSS + 1]
    Yer = run_sr_tsne(Ytsne, Pcoo.row.astype(np.int64), Pcoo.col.astype(np.int64),
                      Pcoo.data.astype(np.float64), nidx, lambda_dlp=LAM, n_iter=1000)
    np.save(f'{CACHE}/{tag}_srtsne_lam{LAM}.npy', Yer)
    return n


if __name__ == '__main__':
    tags = sys.argv[1:] or [os.path.basename(f)[:-6] for f in sorted(glob.glob(f'{DATA}/*_n.csv'))]
    for i, t in enumerate(tags):
        t0 = time.time()
        n = embed(t)
        print(f'[{i+1}/{len(tags)}] {t}: n={n} done {time.time()-t0:.0f}s', flush=True)
    print('ALL DONE')
