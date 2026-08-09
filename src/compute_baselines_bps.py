"""Method x cost table: mean net return over 35 tickers at 0/0.2/0.5/1/2bp, plus wins vs B&H.
Ours and random recompute markers at each cost via the global P* rule; the classical signals do not depend on cost, only the charge does."""
import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import insample_fixP as F
from compute_baselines import load, equity, sig_ma, sig_rsi, sig_mom, sig_random, STOCKS

COSTS = [0, 0.2, 0.5, 1, 2]
PSTAR = {0: 80, 0.2: 80, 0.5: 50, 1: 40, 2: 10}
METHODS = ['B&H', 'MA(5/20)', 'RSI(14)', 'Momentum', 'Random', 'Original', 'ours']
R = lambda eqv: (eqv[-1] - 10000) / 100

ret = {m: {c: [] for c in COSTS} for m in METHODS}
win = {m: {c: 0 for c in COSTS} for m in METHODS}
for ti, t in enumerate(STOCKS):
    X, y, prices, Yt, Ye = load(t); up, dn = (y == 1), (y == -1)
    bh_ret = (prices[-1] - prices[0]) / prices[0] * 100
    mg = F.dbscan_markers(Yt)
    dma, uma = sig_ma(prices); drsi, ursi = sig_rsi(prices); dmo, umo = sig_mom(prices)
    for c in COSTS:
        mo = (F.dbscan_k_v2(Ye, X, k=4, minPts=30, outlier_top_pct=PSTAR[c], labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
        v = {'B&H': bh_ret,
             'MA(5/20)': R(equity(uma, dma, prices, c)),
             'RSI(14)': R(equity(ursi, drsi, prices, c)),
             'Momentum': R(equity(umo, dmo, prices, c)),
             'Original': R(equity(mg & up, mg & dn, prices, c)),
             'ours': R(equity(mo & up, mo & dn, prices, c))}
        nb, ns = int((mo & dn).sum()), int((mo & up).sum())
        v['Random'] = np.mean([R(equity(*sig_random(nb, ns, len(prices), np.random.default_rng(1000*ti + 100*int(c*10) + s))[::-1], prices, c)) for s in range(20)])
        for m in METHODS:
            ret[m][c].append(v[m])
            if v[m] > bh_ret: win[m][c] += 1

print('=' * 62)
print('Method x cost: mean net return (%) over 35 tickers')
print(f'{"method":12}' + ''.join(f'{str(c)+"bp":>9}' for c in COSTS))
for m in METHODS:
    print(f'{m:12}' + ''.join(f'{np.mean(ret[m][c]):>9.1f}' for c in COSTS))
print('\nWins vs B&H (/35)')
print(f'{"method":12}' + ''.join(f'{str(c)+"bp":>9}' for c in COSTS))
for m in METHODS:
    print(f'{m:12}' + ''.join(f'{str(win[m][c])+"/35":>9}' for c in COSTS))
