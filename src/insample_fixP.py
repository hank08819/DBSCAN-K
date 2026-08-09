"""In-sample grid over the filter level P (sixteen-stock pilot).

Runs the full (ticker, embedding, method, P) x cost grid on the pilot stocks
with one shared hyper-parameter setting: minPts=30, dlp_k=10, dlp_weight=1.0.
Embeddings are read from cache/. No train/test split; see
src/walkforward_eval.py for the out-of-sample protocol and
src/select_pstar_pilot.py for how P*(c) is read off this grid.

The thirty-five-stock panel reported in the paper is produced by
src/grid_panel.py and src/compute_baselines*.py, which use lambda = 1.0 for
every ticker.

Output: the full grid as CSV under results/.
"""
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

DATA_DIR = 'data'
CACHE_DIR = 'cache'
OUT_DIR = 'results'

# per-ticker SR-tSNE lambda (pilot only; AAPL 2.0, all others 1.0)
STOCK_LAMBDA = {
    'AAPL': 2.0, 'AEO': 1.0, 'AMZN': 1.0, 'BAC': 1.0, 'CPRI': 1.0,
    'GES': 1.0, 'GOOG': 1.0, 'GS': 1.0, 'HD': 1.0, 'HSBC': 1.0,
    'JCP': 1.0, 'JPM': 1.0, 'MSFT': 1.0, 'TGT': 1.0, 'TIF': 1.0, 'WMT': 1.0,
}
ALL_STOCKS = ['AAPL', 'AEO', 'AMZN', 'BAC', 'CPRI', 'GES', 'GOOG', 'GS',
              'HD', 'HSBC', 'JCP', 'JPM', 'MSFT', 'TGT', 'TIF', 'WMT']

COST_LEVELS = [0, 0.2, 0.5, 1, 2, 5, 10, 20]  # includes 0.2bp; realistic US large-cap spreads are ~0.2-2bp
SWEEP_PCTS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # includes very small P: purest tail signal, fewest trades

# shared hyper-parameters, identical for all sixteen pilot tickers
DEF_MINPTS = 30
DEF_DLP_K = 10
DEF_DLP_W = 1.0

# load dbscan_k_v2
_spec = importlib.util.spec_from_file_location(
    'dbscan_k_v2', os.path.join(os.path.dirname(__file__), 'dbscan_k_v2.py'))
_v2mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2mod)
dbscan_k_v2 = _v2mod.dbscan_k_v2


# ---------------------------------------------------------------------------
# down-marker -> buy, up-marker -> sell, all-in/all-out, cost charged on entry and exit
# ---------------------------------------------------------------------------
def run_trading(up_mask, down_mask, prices, cost_bps=0):
    cost_rate = cost_bps / 10000 / 2
    cash, pos, state = 10000.0, 0.0, 'C'
    trades = 0
    for i in range(len(prices)):
        p = prices[i]
        if down_mask[i] and state == 'C' and cash > 0:
            cash -= cash * cost_rate; pos = cash / p; cash = 0; state = 'P'; trades += 1
        elif up_mask[i] and state == 'P' and pos > 0:
            cash = pos * p; cash -= cash * cost_rate; pos = 0; state = 'C'; trades += 1
    if state == 'P':
        cash = pos * prices[-1]
    return (cash - 10000) / 100, trades


# reads <tag>_n.csv without dropna so row i matches cache row i
def load_stock(tag):
    df = pd.read_csv(f'{DATA_DIR}/{tag}_n.csv')
    df['labels'] = np.where(df['marketClose'] >= df['marketOpen'], 1, -1)
    drop_cols = [c for c in df.columns if 'Unnamed' in c or c == 'Date']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_raw = df[feature_cols].values.astype(np.float64)
    y = df['labels'].values
    prices = df['marketClose'].values.astype(float)
    return X_raw, y, prices


# plain DBSCAN: every noise point is a marker, no P sweep
def dbscan_markers(Y, minPts=30, eps_pct=25):
    nn = NearestNeighbors(n_neighbors=minPts).fit(Y)
    eps = np.percentile(nn.kneighbors(Y)[0][:, -1], eps_pct)
    return DBSCAN(eps=eps, min_samples=minPts).fit_predict(Y) == -1


# ---------------------------------------------------------------------------
# full in-sample grid for one ticker
# ---------------------------------------------------------------------------
def eval_stock(tag, verbose=True):
    lam = STOCK_LAMBDA[tag]
    X_raw, y, prices = load_stock(tag)
    X_scaled = StandardScaler().fit_transform(X_raw)   # fit on the whole segment; this is X_high for dbscan_k_v2
    is_up, is_down = (y == 1), (y == -1)
    buyhold = (prices[-1] - prices[0]) / prices[0] * 100

    Y = {'tSNE': np.load(f'{CACHE_DIR}/{tag}_tsne.npy'),
         'SR-tSNE': np.load(f'{CACHE_DIR}/{tag}_srtsne_lam{lam}.npy')}
    assert Y['tSNE'].shape[0] == len(prices), f'{tag} tsne row count mismatch'
    assert Y['SR-tSNE'].shape[0] == len(prices), f'{tag} er row count mismatch'

    rows = []
    for emb, Yemb in Y.items():
        # DBSCAN+K: sweep over all P with shared hyper-parameters
        for P in SWEEP_PCTS:
            R = dbscan_k_v2(Yemb, X_scaled, k=4, minPts=DEF_MINPTS,
                            outlier_top_pct=P, labels_dir=y,
                            dlp_k=DEF_DLP_K, dlp_weight=DEF_DLP_W)
            mk = (R == 4)
            row = {'tag': tag, 'emb': emb, 'method': 'K_v2', 'P': P,
                   'n_markers': int(mk.sum()), 'buyhold': round(buyhold, 3), 'lambda': lam}
            for c in COST_LEVELS:
                row[f'ret_{c}bp'] = round(run_trading(mk & is_up, mk & is_down, prices, c)[0], 3)
            rows.append(row)
        # plain DBSCAN baseline, no P
        mkp = dbscan_markers(Yemb)
        row = {'tag': tag, 'emb': emb, 'method': 'plain', 'P': None,
               'n_markers': int(mkp.sum()), 'buyhold': round(buyhold, 3), 'lambda': lam}
        for c in COST_LEVELS:
            row[f'ret_{c}bp'] = round(run_trading(mkp & is_up, mkp & is_down, prices, c)[0], 3)
        rows.append(row)
        if verbose:
            best = max(r['ret_0bp'] for r in rows if r['emb'] == emb and r['method'] == 'K_v2')
            print(f'  [{tag}] {emb}: K_v2 best@0bp over P = {best:+.2f}%  '
                  f'(buyhold {buyhold:+.2f}%)', flush=True)
    return rows


def run_all(stocks=None, out_csv='grid.csv'):
    import time
    stocks = stocks or ALL_STOCKS
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    for i, tag in enumerate(stocks):
        t0 = time.time()
        print(f'\n>>> {tag} ({i+1}/{len(stocks)})', flush=True)
        all_rows.extend(eval_stock(tag))
        print(f'<<< {tag} done {time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(f'{OUT_DIR}/{out_csv}', index=False)
    print(f'\n[saved] {OUT_DIR}/{out_csv} ({len(df)} rows)', flush=True)
    return df


# ---------------------------------------------------------------------------
def _self_test():
    print('=== insample_fixP SELF-TEST ===')
    prices = np.array([10., 11., 12., 13., 14.])
    up = np.array([False, False, False, False, True])
    down = np.array([True, False, False, False, False])
    ret, tr = run_trading(up, down, prices, 0)
    assert abs(ret - 40.0) < 1e-6, f'expected 40%, got {ret}'
    assert tr == 2
    ret_c, _ = run_trading(up, down, prices, 100)
    assert ret_c < ret
    ret0, tr0 = run_trading(np.zeros(5, bool), np.zeros(5, bool), prices, 0)
    assert ret0 == 0.0 and tr0 == 0
    print(f'  [ok] run_trading: 40% / cost applied / no markers -> 0')
    assert len(ALL_STOCKS) == 16 and set(STOCK_LAMBDA) == set(ALL_STOCKS)
    print('=== PASSED ===\n')


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if arg == 'selftest':
        _self_test()
    elif arg == 'all':
        _self_test()
        run_all()
    else:
        _self_test()
        rows = eval_stock(arg)
        df = pd.DataFrame(rows)
        cols = ['emb', 'method', 'P', 'n_markers'] + [f'ret_{c}bp' for c in COST_LEVELS]
        print(f'\n{arg} (λ={STOCK_LAMBDA[arg]})  buyhold={rows[0]["buyhold"]:+.2f}%')
        print(df[cols].to_string(index=False))
