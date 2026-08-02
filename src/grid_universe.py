"""扩池 in-sample grid, 可分片并行 (纯 CPU)。逐列对齐 grid_new_stocks.py 的 schema。
用法: python grid_universe.py <idx> <total>   -> 写 result/insample_fixP/grid_part{idx}.csv
只处理有 ER-tSNE cache 的 tag; 无 cache 的跳过。chain 跑完 concat 成 grid_new.csv。"""
import sys, glob, os, time
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import insample_fixP as F

DATA = 'data'
CACHE = 'cache'
OUT = 'results'
COSTS = [0, 0.2, 0.5, 1, 2]
PS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]
LAM = 1.0


def load(tag):
    df = pd.read_csv(f'{DATA}/{tag}_n.csv')
    y = np.where(df.marketClose >= df.marketOpen, 1, -1)
    cols = [c for c in df.columns if c != 'Date']
    X = StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    prices = df.marketClose.values.astype(float)
    return X, y, prices


def eval_stock(tag):
    X, y, prices = load(tag)
    is_up, is_down = (y == 1), (y == -1)
    bh = (prices[-1] - prices[0]) / prices[0] * 100
    Yemb = np.load(f'{CACHE}/{tag}_ertsne_lam{LAM}.npy')
    assert Yemb.shape[0] == len(prices), f'{tag} 行数不齐'
    rows = []
    for P in PS:
        R = F.dbscan_k_v2(Yemb, X, k=4, minPts=30, outlier_top_pct=P,
                          labels_dir=y, dlp_k=10, dlp_weight=1.0)
        mk = (R == 4)
        row = {'tag': tag, 'emb': 'ER-tSNE', 'method': 'K_v2', 'P': P,
               'n_markers': int(mk.sum()), 'buyhold': round(bh, 3), 'lambda': LAM}
        for c in COSTS:
            row[f'ret_{c}bp'] = round(F.run_trading(mk & is_up, mk & is_down, prices, c)[0], 3)
        rows.append(row)
    mkp = F.dbscan_markers(Yemb)
    row = {'tag': tag, 'emb': 'ER-tSNE', 'method': 'plain', 'P': None,
           'n_markers': int(mkp.sum()), 'buyhold': round(bh, 3), 'lambda': LAM}
    for c in COSTS:
        row[f'ret_{c}bp'] = round(F.run_trading(mkp & is_up, mkp & is_down, prices, c)[0], 3)
    rows.append(row)
    return rows


if __name__ == '__main__':
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    all_tags = sorted(os.path.basename(f)[:-6] for f in glob.glob(f'{DATA}/*_n.csv'))
    tags = [t for t in all_tags[idx::total]
            if os.path.exists(f'{CACHE}/{t}_ertsne_lam{LAM}.npy')]
    allrows = []
    for i, t in enumerate(tags):
        t0 = time.time()
        try:
            allrows.extend(eval_stock(t))
            print(f'[{idx}/{total} {i+1}/{len(tags)}] {t} {time.time()-t0:.0f}s', flush=True)
        except Exception as e:
            print(f'[{idx}/{total} {i+1}/{len(tags)}] {t} FAIL {repr(e)[:80]}', flush=True)
    pd.DataFrame(allrows).to_csv(f'{OUT}/grid_part{idx}.csv', index=False)
    print(f'[{idx}/{total}] saved grid_part{idx}.csv {len(allrows)} rows', flush=True)
