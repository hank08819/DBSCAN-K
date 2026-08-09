"""[walk-forward step 1] Independent embeddings for the 40/20 day split.

Each ticker is cut into a 40-day fitting window and a 20-day test window, and
each window is embedded on its own (t-SNE + SR-tSNE, lambda=1.0). t-SNE is
transductive, so slicing a full-window embedding in time would let test-period
bars take part in the embedding. StandardScaler is also fit within each window.

Writes cache/{tag}_wf{train,test}_{tsne,srtsne_lam1.0}.npy.
Run from the repository root:  python src/embed_walkforward.py
"""
import sys, time, os
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
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
K_LOSS, LAM, TRAIN_DAYS = 10, 1.0, 40
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
          'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']


def embed_window(X, tag, win):
    n = X.shape[0]
    Ytsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000,
                 method='barnes_hut', learning_rate='auto', init='pca').fit_transform(X)
    np.save(f'{CACHE}/{tag}_wf{win}_tsne.npy', Ytsne)
    Dsq = pairwise_distances(X, metric='sqeuclidean')
    Pc = _joint_probabilities(Dsq, 30, 0)
    Pd = squareform(Pc); np.fill_diagonal(Pd, 0.0)
    Pcoo = coo_matrix(Pd); del Pd, Dsq
    nidx = NearestNeighbors(n_neighbors=K_LOSS + 1, n_jobs=-1).fit(X).kneighbors(
        return_distance=False)[:, 1:K_LOSS + 1]
    Yer = run_sr_tsne(Ytsne, Pcoo.row.astype(np.int64), Pcoo.col.astype(np.int64),
                      Pcoo.data.astype(np.float64), nidx, lambda_dlp=LAM, n_iter=1000)
    np.save(f'{CACHE}/{tag}_wf{win}_srtsne_lam{LAM}.npy', Yer)
    return n


if __name__ == '__main__':
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    todo = only if only else STOCKS
    for i, tag in enumerate(todo):
        done = all(os.path.exists(f'{CACHE}/{tag}_wf{w}_{s}.npy')
                   for w in ('train', 'test')
                   for s in ('tsne', f'srtsne_lam{LAM}'))
        if done:
            print(f'[{i+1}/{len(todo)}] {tag}: cached, skip', flush=True); continue
        t0 = time.time()
        df = pd.read_csv(f'{DATA}/{tag}_n.csv')
        days = pd.to_datetime(df.Date).dt.date
        udays = sorted(days.unique())
        tr_mask = days.isin(udays[:TRAIN_DAYS]).values
        cols = [c for c in df.columns if c != 'Date']
        raw = df[cols].values.astype(np.float64)
        ns = []
        for win, mask in [('train', tr_mask), ('test', ~tr_mask)]:
            Xw = StandardScaler().fit_transform(raw[mask])
            ns.append(embed_window(Xw, tag, win))
        print(f'[{i+1}/{len(todo)}] {tag}: train n={ns[0]} test n={ns[1]} '
              f'done {time.time()-t0:.0f}s', flush=True)
    print('ALL DONE')
