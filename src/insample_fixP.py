"""
In-sample 固定 P 评估 (导师认可 in-sample 口径)。

目的: 去掉原主表的 per-stock selection (挑 P / 挑 λ / 挑 tSNE-vs-ER / tune 6 只 4D grid),
      统一超参, 只让 P 固定 (方案1) 或按成本规则 (方案2), 看还能 beat FIDR-SCAN 几只。

协议:
  - 全 5850 bar in-sample (不切 train/test), 与原始 <STOCK>_v2.ipynb 口径一致
  - embedding 用现成 cache: DLP/data/cache/{tag}_tsne.npy + {tag}_ertsne_lam{λ}.npy
    (已核对: 两 cache 与 csv 均 5850 行, dropna 不掉行 -> 行 i 对齐 csv 行 i)
  - 统一超参: 全 16 只 minPts=30, dlp_k=10, dlp_weight=1.0 (原 v2 default)
             tune 6 只不再 4D grid (那也是 selection)
  - 输出全网格 CSV: (tag, emb, method, P) × cost 的 in-sample return
    方案1a/1b/2 由 analyze_insample.py 从网格切, 不重复跑

复用 b1_eval.py: run_trading / load_stock / dbscan_markers (verbatim, 见下注明)。
dbscan_k_v2 用 spec load (与 b1_eval 同法)。
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

# 每只 ER-tSNE λ (来自 run_ertsne_all.py; AAPL 2.0 其余 1.0)
STOCK_LAMBDA = {
    'AAPL': 2.0, 'AEO': 1.0, 'AMZN': 1.0, 'BAC': 1.0, 'CPRI': 1.0,
    'GES': 1.0, 'GOOG': 1.0, 'GS': 1.0, 'HD': 1.0, 'HSBC': 1.0,
    'JCP': 1.0, 'JPM': 1.0, 'MSFT': 1.0, 'TGT': 1.0, 'TIF': 1.0, 'WMT': 1.0,
}
ALL_STOCKS = ['AAPL', 'AEO', 'AMZN', 'BAC', 'CPRI', 'GES', 'GOOG', 'GS',
              'HD', 'HSBC', 'JCP', 'JPM', 'MSFT', 'TGT', 'TIF', 'WMT']

COST_LEVELS = [0, 0.2, 0.5, 1, 2, 5, 10, 20]  # 加 0.2bp 超低成本档 (现实美股大盘股点差 ~0.2-2bp)
SWEEP_PCTS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]  # 加极小 P (极端尾部信号最纯+交易最少)

# 统一超参 (原 v2 default, 全 16 只一致, 不再 per-stock tune)
DEF_MINPTS = 30
DEF_DLP_K = 10
DEF_DLP_W = 1.0

# 加载 dbscan_k_v2 (与 b1_eval 同法)
_spec = importlib.util.spec_from_file_location(
    'dbscan_k_v2', os.path.join(os.path.dirname(__file__), 'dbscan_k_v2.py'))
_v2mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2mod)
dbscan_k_v2 = _v2mod.dbscan_k_v2


# ---------------------------------------------------------------------------
# verbatim from b1_eval.py (down->buy, up->sell, all-in/all-out, 开+平各扣一次)
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


# verbatim from b1_eval.py (读 <tag>_n.csv, 不 dropna -> 与 cache 对齐已核对)
def load_stock(tag):
    df = pd.read_csv(f'{DATA_DIR}/{tag}_n.csv')
    df['labels'] = np.where(df['marketClose'] >= df['marketOpen'], 1, -1)
    drop_cols = [c for c in df.columns if 'Unnamed' in c or c == 'Date']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_raw = df[feature_cols].values.astype(np.float64)
    y = df['labels'].values
    prices = df['marketClose'].values.astype(float)
    return X_raw, y, prices


# verbatim from b1_eval.py (plain DBSCAN 全噪声点当 marker, 无 P sweep)
def dbscan_markers(Y, minPts=30, eps_pct=25):
    nn = NearestNeighbors(n_neighbors=minPts).fit(Y)
    eps = np.percentile(nn.kneighbors(Y)[0][:, -1], eps_pct)
    return DBSCAN(eps=eps, min_samples=minPts).fit_predict(Y) == -1


# ---------------------------------------------------------------------------
# 单只 in-sample 全网格
# ---------------------------------------------------------------------------
def eval_stock(tag, verbose=True):
    lam = STOCK_LAMBDA[tag]
    X_raw, y, prices = load_stock(tag)
    X_scaled = StandardScaler().fit_transform(X_raw)   # 全段 fit, 给 dbscan_k_v2 的 X_high
    is_up, is_down = (y == 1), (y == -1)
    buyhold = (prices[-1] - prices[0]) / prices[0] * 100

    Y = {'tSNE': np.load(f'{CACHE_DIR}/{tag}_tsne.npy'),
         'ER-tSNE': np.load(f'{CACHE_DIR}/{tag}_ertsne_lam{lam}.npy')}
    assert Y['tSNE'].shape[0] == len(prices), f'{tag} tsne 行数不齐'
    assert Y['ER-tSNE'].shape[0] == len(prices), f'{tag} er 行数不齐'

    rows = []
    for emb, Yemb in Y.items():
        # DBSCAN+K_v2: 全 P sweep (统一超参, 不 tune)
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
        # plain DBSCAN baseline (成本穿越对手, 无 P)
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
    print(f'  [ok] run_trading: 40% / cost减 / 无marker=0')
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
