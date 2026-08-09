"""DLP-weight ablation (w=0 pure geometry / 0.5 / 1 as in the paper / 2).
35 tickers x 4 values of w x 12 values of P x 5 cost levels, from cached embeddings, CPU only.
Each w selects its own P*(c) by the same global rule, so the comparison is fair.
Writes results/grid_ablation_w.csv; summary table on stdout."""
import sys, time
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import insample_fixP as F

DATA = 'data'
CACHE = 'cache'
OUT = 'results'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
          'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
COSTS = [0, 0.2, 0.5, 1, 2]
PS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]
WEIGHTS = [0.0, 0.5, 1.0, 2.0]


def load(tag):
    df = pd.read_csv(f'{DATA}/{tag}_n.csv')
    y = np.where(df.marketClose >= df.marketOpen, 1, -1)
    cols = [c for c in df.columns if c != 'Date']
    X = StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    return X, y, df.marketClose.values.astype(float)


rows = []
for i, tag in enumerate(STOCKS):
    t0 = time.time()
    X, y, pr = load(tag)
    up, dn = (y == 1), (y == -1)
    bh = (pr[-1] - pr[0]) / pr[0] * 100
    Yemb = np.load(f'{CACHE}/{tag}_srtsne_lam1.0.npy')
    assert Yemb.shape[0] == len(pr), f'{tag} row count mismatch'
    for w in WEIGHTS:
        for P in PS:
            R = F.dbscan_k_v2(Yemb, X, k=4, minPts=30, outlier_top_pct=P,
                              labels_dir=y, dlp_k=10, dlp_weight=w)
            mk = (R == 4)
            row = {'tag': tag, 'w': w, 'P': P, 'n_markers': int(mk.sum()), 'buyhold': round(bh, 3)}
            for c in COSTS:
                row[f'ret_{c}bp'] = round(F.run_trading(mk & up, mk & dn, pr, c)[0], 3)
            rows.append(row)
    print(f'[{i+1}/{len(STOCKS)}] {tag} done {time.time()-t0:.0f}s', flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/grid_ablation_w.csv', index=False)
print(f'[saved] {OUT}/grid_ablation_w.csv ({len(df)} rows)')

# summary: each w picks its own P*(c); report mean return and wins vs buy-and-hold
def bhc(bh, c): return ((1 - c / 20000) * (bh / 100 + 1) - 1) * 100
print(f'\n{"w":>4} {"cost":>5} {"P*":>4} {"mean_ret":>9} {"wins_vs_BH":>11}')
for w in WEIGHTS:
    dw = df[df.w == w]
    for c in COSTS:
        m = dw.groupby('P')[f'ret_{c}bp'].mean()
        pstar = m.idxmax()
        sel = dw[dw.P == pstar]
        wins = int((sel[f'ret_{c}bp'].values > bhc(sel['buyhold'].values, c)).sum())
        print(f'{w:>4} {c:>5} {int(pstar):>4} {m.max():>9.2f} {wins:>8}/35')
