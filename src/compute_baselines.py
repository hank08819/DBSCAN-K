"""Adds four classical timing baselines to B&H, Original and ours: MA crossover, RSI, momentum, random.
All 35 tickers at 0.5bp through the same all-in/all-out engine. Reports Return%, Sharpe, MaxDD, wins.
Signal convention: down_mask = buy (enter), up_mask = sell (exit)."""
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
    peak = np.maximum.accumulate(eqv); dd = (eqv - peak) / peak
    return [(eqv[-1] - 10000) / 100, sharpe, -dd.min() * 100]   # ret%, Sharpe, maxDD%


# ---- baseline signals: return (down_mask = buy, up_mask = sell) ----
def sig_ma(prices, s=5, l=20):
    p = pd.Series(prices); ms = p.rolling(s).mean().values; ml = p.rolling(l).mean().values
    n = len(prices); dn = np.zeros(n, bool); up = np.zeros(n, bool)
    for i in range(1, n):
        if np.isnan(ms[i]) or np.isnan(ml[i]) or np.isnan(ms[i-1]) or np.isnan(ml[i-1]): continue
        if ms[i] > ml[i] and ms[i-1] <= ml[i-1]: dn[i] = True   # fast crosses above slow -> buy
        elif ms[i] < ml[i] and ms[i-1] >= ml[i-1]: up[i] = True  # fast crosses below slow -> sell
    return dn, up


def sig_rsi(prices, w=14):
    p = pd.Series(prices); d = p.diff()
    gain = d.clip(lower=0).rolling(w).mean(); loss = (-d.clip(upper=0)).rolling(w).mean()
    rsi = (100 - 100 / (1 + gain / (loss + 1e-12))).values
    return (rsi < 30), (rsi > 70)   # buy oversold, sell overbought; nan comparisons are False


def sig_mom(prices, n=20):
    p = np.asarray(prices); r = np.full(len(p), np.nan); r[n:] = p[n:] / p[:-n] - 1
    return (r > 0), (r < 0)   # hold while momentum is positive, exit once negative


def sig_random(n_buy, n_sell, n, rng):
    dn = np.zeros(n, bool); up = np.zeros(n, bool)
    idx = rng.choice(n, n_buy + n_sell, replace=False)
    dn[idx[:n_buy]] = True; up[idx[n_buy:]] = True
    return dn, up


rows = {k: [] for k in ['B&H', 'MA(5/20)', 'RSI(14)', 'Momentum', 'Random', 'Original', 'ours']}
wins = {k: 0 for k in rows}
for ti, t in enumerate(STOCKS):
    X, y, prices, Yt, Ye = load(t); up, dn = (y == 1), (y == -1)
    bh_ret = (prices[-1] - prices[0]) / prices[0] * 100
    mo = (F.dbscan_k_v2(Ye, X, k=4, minPts=30, outlier_top_pct=P50, labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
    mg = F.dbscan_markers(Yt)

    def add(name, eqv):
        r = risk(eqv); rows[name].append(r)
        if r[0] > bh_ret: wins[name] += 1

    add('B&H', bh_eq(prices))
    d, u = sig_ma(prices);  add('MA(5/20)', equity(u, d, prices, COST))
    d, u = sig_rsi(prices); add('RSI(14)', equity(u, d, prices, COST))
    d, u = sig_mom(prices); add('Momentum', equity(u, d, prices, COST))
    nb, ns = int((mo & dn).sum()), int((mo & up).sum())      # random control matches our buy/sell counts
    rr = []
    for s in range(20):
        rng = np.random.default_rng(1000 * ti + s)
        d, u = sig_random(nb, ns, len(prices), rng); rr.append(risk(equity(u, d, prices, COST)))
    rmean = np.mean(rr, 0); rows['Random'].append(rmean)
    if rmean[0] > bh_ret: wins['Random'] += 1
    add('Original', equity(mg & up, mg & dn, prices, COST))
    add('ours', equity(mo & up, mo & dn, prices, COST))

print('=' * 58)
print('Baselines at 0.5bp (mean over 35 tickers, Sharpe annualized)')
print(f'{"method":12}{"Return%":>9}{"Sharpe":>9}{"MaxDD%":>9}{"Wins":>8}')
for k in ['B&H', 'MA(5/20)', 'RSI(14)', 'Momentum', 'Random', 'Original', 'ours']:
    a = np.array(rows[k]); mu = a.mean(0)
    print(f'{k:12}{mu[0]:>9.1f}{mu[1]:>9.2f}{mu[2]:>9.1f}{wins[k]:>6}/35')
