"""在 B&H + Original + ours 之上, 加 4 个经典择时 baseline: MA交叉 / RSI / Momentum / Random。
全部 35 只, 0.5bp, 套现有 all-in/all-out equity() 框架, 现算净值。报 Return%/Sharpe/MaxDD/Wins。
信号约定: down_mask=买入(进场), up_mask=卖出(离场), 同 equity() 里的语义。"""
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
    peak = np.maximum.accumulate(eqv); dd = (eqv - peak) / peak
    return [(eqv[-1] - 10000) / 100, sharpe, -dd.min() * 100]   # ret%, Sharpe, maxDD%


# ---- baseline 信号: 返回 (down_mask 买, up_mask 卖) ----
def sig_ma(prices, s=5, l=20):
    p = pd.Series(prices); ms = p.rolling(s).mean().values; ml = p.rolling(l).mean().values
    n = len(prices); dn = np.zeros(n, bool); up = np.zeros(n, bool)
    for i in range(1, n):
        if np.isnan(ms[i]) or np.isnan(ml[i]) or np.isnan(ms[i-1]) or np.isnan(ml[i-1]): continue
        if ms[i] > ml[i] and ms[i-1] <= ml[i-1]: dn[i] = True   # 金叉 -> 买
        elif ms[i] < ml[i] and ms[i-1] >= ml[i-1]: up[i] = True  # 死叉 -> 卖
    return dn, up


def sig_rsi(prices, w=14):
    p = pd.Series(prices); d = p.diff()
    gain = d.clip(lower=0).rolling(w).mean(); loss = (-d.clip(upper=0)).rolling(w).mean()
    rsi = (100 - 100 / (1 + gain / (loss + 1e-12))).values
    return (rsi < 30), (rsi > 70)   # 超卖买 / 超买卖 (nan 比较自动 False)


def sig_mom(prices, n=20):
    p = np.asarray(prices); r = np.full(len(p), np.nan); r[n:] = p[n:] / p[:-n] - 1
    return (r > 0), (r < 0)   # 动量为正持有 / 转负离场


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
    nb, ns = int((mo & dn).sum()), int((mo & up).sum())      # random 匹配 ours 的买/卖信号数
    rr = []
    for s in range(20):
        rng = np.random.default_rng(1000 * ti + s)
        d, u = sig_random(nb, ns, len(prices), rng); rr.append(risk(equity(u, d, prices, COST)))
    rmean = np.mean(rr, 0); rows['Random'].append(rmean)
    if rmean[0] > bh_ret: wins['Random'] += 1
    add('Original', equity(mg & up, mg & dn, prices, COST))
    add('ours', equity(mo & up, mo & dn, prices, COST))

print('=' * 58)
print('Baseline 对比 @0.5bp (35 只均值, Sharpe/年化)')
print(f'{"method":12}{"Return%":>9}{"Sharpe":>9}{"MaxDD%":>9}{"Wins":>8}')
for k in ['B&H', 'MA(5/20)', 'RSI(14)', 'Momentum', 'Random', 'Original', 'ours']:
    a = np.array(rows[k]); mu = a.mean(0)
    print(f'{k:12}{mu[0]:>9.1f}{mu[1]:>9.2f}{mu[2]:>9.1f}{wins[k]:>6}/35')
