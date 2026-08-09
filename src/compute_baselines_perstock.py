"""Per-ticker baselines: net return, Sharpe and MaxDD for 35 tickers x 7 methods x 5 cost levels.
Same logic and seeds as compute_baselines_bps.py, so the means reproduce tab_baseline.
Output: results/baselines_perstock35.csv (long format: ticker, cost_bp, method, ret, sharpe, maxdd)"""
import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import insample_fixP as F
from compute_baselines import load, equity, risk, bh_eq, sig_ma, sig_rsi, sig_mom, sig_random, STOCKS

COSTS = [0, 0.2, 0.5, 1, 2]
PSTAR = {0: 80, 0.2: 80, 0.5: 50, 1: 40, 2: 10}
METHODS = ['B&H', 'MA(5/20)', 'RSI(14)', 'Momentum', 'Random', 'Original', 'ours']
OUT = 'results/baselines_perstock35.csv'

recs = []
for ti, t in enumerate(STOCKS):
    X, y, prices, Yt, Ye = load(t); up, dn = (y == 1), (y == -1)
    mg = F.dbscan_markers(Yt)
    dma, uma = sig_ma(prices); drsi, ursi = sig_rsi(prices); dmo, umo = sig_mom(prices)
    for c in COSTS:
        # ours recomputes markers per cost; classical signals are cost-independent; B&H pays one entry cost
        mo = (F.dbscan_k_v2(Ye, X, k=4, minPts=30, outlier_top_pct=PSTAR[c], labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
        eqs = {'B&H': bh_eq(prices),
               'MA(5/20)': equity(uma, dma, prices, c),
               'RSI(14)': equity(ursi, drsi, prices, c),
               'Momentum': equity(umo, dmo, prices, c),
               'Original': equity(mg & up, mg & dn, prices, c),
               'ours': equity(mo & up, mo & dn, prices, c)}
        for m, eqv in eqs.items():
            r = risk(eqv)
            recs.append({'ticker': t, 'cost_bp': c, 'method': m, 'ret': r[0], 'sharpe': r[1], 'maxdd': r[2]})
        # random control: matches our buy/sell counts at that cost, averaged over 20 seeds
        nb, ns = int((mo & dn).sum()), int((mo & up).sum())
        rr = []
        for s in range(20):
            rng = np.random.default_rng(1000 * ti + 100 * int(c * 10) + s)
            d, u = sig_random(nb, ns, len(prices), rng)
            rr.append(risk(equity(u, d, prices, c)))
        rmean = np.mean(rr, 0)
        recs.append({'ticker': t, 'cost_bp': c, 'method': 'Random', 'ret': rmean[0], 'sharpe': rmean[1], 'maxdd': rmean[2]})
    print(f'[{ti+1:2d}/35] {t} done', flush=True)

df = pd.DataFrame(recs)
df.to_csv(OUT, index=False)
print(f'\nsaved {len(df)} rows -> {OUT}')

# cross-check: these means should match tab_baseline.tex
piv = df.pivot_table(index='method', columns='cost_bp', values='ret').reindex(METHODS)
print('\ncross-check against tab_baseline:')
print(piv.round(1).to_string())
# wins vs B&H
bh = df[df.method == 'B&H'].set_index(['ticker', 'cost_bp']).ret
w = df[df.method != 'B&H'].copy()
w['win'] = w.apply(lambda r: r.ret > bh.loc[(r.ticker, r.cost_bp)], axis=1)
print('\nWins vs B&H (/35):')
print(w.pivot_table(index='method', columns='cost_bp', values='win', aggfunc='sum').reindex(METHODS[1:]).astype(int).to_string())
