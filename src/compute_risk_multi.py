"""Risk metrics (net return / Sharpe / vol / max drawdown) for ours, the original
detector, and buy-and-hold over the 35 equities at 0, 0.5, and 2 bp (paper Table 6).
The 0.5 bp rows cross-check the numbers printed by compute_risk_ablation.py.
P follows the frozen global schedule {0: 80, 0.5: 50, 2: 10}; annualization sqrt(78*252).
Writes results/risk_multi.csv."""
import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import insample_fixP as F

DATA_EXT = 'data'; CACHE = 'cache'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
          'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
PSTAR = {0.0: 80, 0.5: 50, 2.0: 10}
ANN = np.sqrt(78 * 252)


def load(t):
    dd = pd.read_csv(f'{DATA_EXT}/{t}_n.csv'); yy = np.where(dd.marketClose >= dd.marketOpen, 1, -1)
    cols = [c for c in dd.columns if c != 'Date']
    X = StandardScaler().fit_transform(dd[cols].values.astype(np.float64))
    return X, yy, dd.marketClose.values.astype(float), np.load(f'{CACHE}/{t}_tsne.npy'), np.load(f'{CACHE}/{t}_srtsne_lam1.0.npy')


def equity(up_mask, down_mask, prices, c):
    rate = c / 10000 / 2; cash, pos, st = 10000.0, 0.0, 'C'; eq = np.empty(len(prices))
    for i in range(len(prices)):
        p = prices[i]
        if down_mask[i] and st == 'C': cash -= cash*rate; pos = cash/p; cash = 0; st = 'P'
        elif up_mask[i] and st == 'P': cash = pos*p; cash -= cash*rate; pos = 0; st = 'C'
        eq[i] = cash if st == 'C' else pos*p
    return eq


def bh_eq(prices): return 10000.0 / prices[0] * prices


def risk(eqv):
    r = np.diff(eqv) / eqv[:-1]
    sharpe = r.mean() / (r.std() + 1e-12) * ANN
    vol = r.std() * ANN * 100
    peak = np.maximum.accumulate(eqv); dd = (eqv - peak) / peak
    return [(eqv[-1] - 10000) / 100, sharpe, vol, -dd.min() * 100]   # ret%, Sharpe, vol%, maxDD%


rows = {c: {'ours': [], 'orig': [], 'B&H': []} for c in PSTAR}
for t in STOCKS:
    X, y, prices, Yt, Ye = load(t); up, dn = (y == 1), (y == -1)
    mg = F.dbscan_markers(Yt)
    for c, P in PSTAR.items():
        mo = (F.dbscan_k_v2(Ye, X, k=4, minPts=30, outlier_top_pct=P, labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
        rows[c]['ours'].append(risk(equity(mo & up, mo & dn, prices, c)))
        rows[c]['orig'].append(risk(equity(mg & up, mg & dn, prices, c)))
        rows[c]['B&H'].append(risk(bh_eq(prices)))
    print(t, flush=True)

out = []
for c in PSTAR:
    for m in ['ours', 'orig', 'B&H']:
        a = np.array(rows[c][m]).mean(axis=0)
        out.append([c, m] + list(a))
        print(f'{c}bp {m:5s} ret {a[0]:6.1f}  sharpe {a[1]:5.2f}  vol {a[2]:5.1f}  maxdd {a[3]:5.1f}')
pd.DataFrame(out, columns=['cost', 'method', 'ret', 'sharpe', 'vol', 'maxdd']).to_csv(
    'results/risk_multi.csv', index=False)
print('saved results/risk_multi.csv')
