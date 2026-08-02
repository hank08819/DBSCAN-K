"""① 风险指标 Sharpe/maxDD/vol (35只, ours/orig/B&H @0.5bp)  ② filter消融 (ER-tSNE上拆dlp/filter)。
净值序列现算; 年化按 5min bar (78 bar/日 x 252 日 = 19656 期/年)。"""
import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import insample_fixP as F

DATA_EXT = 'data'; CACHE = 'cache'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
          'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
COST = 0.5; P50 = 50; ANN = np.sqrt(78 * 252)


def load(t):
    dd = pd.read_csv(f'{DATA_EXT}/{t}_n.csv'); yy = np.where(dd.marketClose >= dd.marketOpen, 1, -1)
    cols = [c for c in dd.columns if c != 'Date']
    X = StandardScaler().fit_transform(dd[cols].values.astype(np.float64))
    return X, yy, dd.marketClose.values.astype(float), np.load(f'{CACHE}/{t}_tsne.npy'), np.load(f'{CACHE}/{t}_ertsne_lam1.0.npy')


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


risk_rows = {'ours': [], 'orig': [], 'B&H': []}
abl = {'full': [], 'no_dlp': [], 'no_filter': [], 'no_both': []}
ablw = {k: 0 for k in abl}
for t in STOCKS:
    X, y, prices, Yt, Ye = load(t); up, dn = (y == 1), (y == -1)
    bh_ret = (prices[-1] - prices[0]) / prices[0] * 100
    mo = (F.dbscan_k_v2(Ye, X, k=4, minPts=30, outlier_top_pct=P50, labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
    mg = F.dbscan_markers(Yt)
    risk_rows['ours'].append(risk(equity(mo & up, mo & dn, prices, COST)))
    risk_rows['orig'].append(risk(equity(mg & up, mg & dn, prices, COST)))
    risk_rows['B&H'].append(risk(bh_eq(prices)))
    for name, (w, P) in {'full': (1.0, P50), 'no_dlp': (0.0, P50), 'no_filter': (1.0, 100), 'no_both': (0.0, 100)}.items():
        mk = (F.dbscan_k_v2(Ye, X, k=4, minPts=30, outlier_top_pct=P, labels_dir=y, dlp_k=10, dlp_weight=w) == 4)
        ret = F.run_trading(mk & up, mk & dn, prices, COST)[0]
        abl[name].append(ret)
        if ret > bh_ret: ablw[name] += 1

# ---- 打印 ----
print('=' * 60)
print('① 风险调整指标 @0.5bp (35只均值, 年化)')
print(f'{"method":8}{"Return%":>9}{"Sharpe":>9}{"Vol%":>9}{"MaxDD%":>9}')
for m in ['ours', 'orig', 'B&H']:
    a = np.array(risk_rows[m]); mu = a.mean(0)
    print(f'{m:8}{mu[0]:>9.1f}{mu[1]:>9.2f}{mu[2]:>9.1f}{mu[3]:>9.1f}')
print()
print('② Filter 消融 @0.5bp (35只, 全在 ER-tSNE, 只改 dlp权重/filter档)')
print(f'{"config":12}{"dlp":>5}{"filter":>8}{"Ret%":>8}{"winB&H":>8}')
desc = {'full': ('1.0', 'P*=50'), 'no_dlp': ('0', 'P*=50'), 'no_filter': ('1.0', 'none(100)'), 'no_both': ('0', 'none(100)')}
for name in ['full', 'no_dlp', 'no_filter', 'no_both']:
    d = desc[name]
    print(f'{name:12}{d[0]:>5}{d[1]:>8}{np.mean(abl[name]):>8.1f}{ablw[name]:>6}/35')
