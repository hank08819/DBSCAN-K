"""[walk-forward step 2] Train-window P* selection and test-window evaluation.

P*(c) is chosen on the fitting window by the same rule as the sixteen-stock
pilot (src/select_pstar_pilot.py): the P that beats buy-and-hold on the most
stocks, with buy-and-hold charged a single one-sided entry cost. Ties are broken
by fitting-window mean net return. The schedule is then frozen and applied to
the test window, against the original detector and buy-and-hold.

The argmax-mean-return criterion is also computed and printed, so the two
selection rules can be compared side by side.

Requires src/embed_walkforward.py to have been run first.
Run from the repository root:  python src/walkforward_eval.py
"""
import sys, os, time, json
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import insample_fixP as F

DATA, CACHE = 'data', 'cache'
OUT = 'results'
os.makedirs(OUT, exist_ok=True)
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
          'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
COSTS = [0, 0.2, 0.5, 1, 2]
PS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TRAIN_DAYS = 40


def load_window(tag, win):
    df = pd.read_csv(f'{DATA}/{tag}_n.csv')
    days = pd.to_datetime(df.Date).dt.date
    udays = sorted(days.unique())
    mask = days.isin(udays[:TRAIN_DAYS]).values
    if win == 'test':
        mask = ~mask
    cols = [c for c in df.columns if c != 'Date']
    X = StandardScaler().fit_transform(df[cols].values.astype(np.float64)[mask])
    y = np.where(df.marketClose >= df.marketOpen, 1, -1)[mask]
    pr = df.marketClose.values.astype(float)[mask]
    Yer = np.load(f'{CACHE}/{tag}_wf{win}_srtsne_lam1.0.npy')
    Yts = np.load(f'{CACHE}/{tag}_wf{win}_tsne.npy')
    assert Yer.shape[0] == len(pr) == Yts.shape[0], f'{tag} {win} row mismatch'
    return X, y, pr, Yer, Yts


def bhc(bh, c):
    """buy-and-hold with a single one-sided entry cost (same as pilot script)."""
    return ((1 - c / 20000) * (np.asarray(bh) / 100 + 1) - 1) * 100


# ---------------- step 1: train grid ----------------
rows = []
for i, tag in enumerate(STOCKS):
    t0 = time.time()
    X, y, pr, Yer, _ = load_window(tag, 'train')
    up, dn = (y == 1), (y == -1)
    bh = (pr[-1] - pr[0]) / pr[0] * 100
    for P in PS:
        mk = (F.dbscan_k_v2(Yer, X, k=4, minPts=30, outlier_top_pct=P,
                            labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
        row = {'tag': tag, 'P': P, 'bh': bh}
        for c in COSTS:
            row[f'ret_{c}bp'] = F.run_trading(mk & up, mk & dn, pr, c)[0]
        rows.append(row)
    print(f'[train {i+1}/{len(STOCKS)}] {tag} {time.time()-t0:.0f}s', flush=True)
tr = pd.DataFrame(rows)
tr.to_csv(f'{OUT}/walkforward_train_grid.csv', index=False)

# ---------------- P*(c): pilot rule = argmax wins vs B&H ----------------
def wins(P, c):
    sub = tr[tr.P == P]
    return int((sub[f'ret_{c}bp'].values > bhc(sub['bh'].values, c)).sum())

PSTAR, PSTAR_MEAN, WINTAB = {}, {}, {}
for c in COSTS:
    w = {P: wins(P, c) for P in PS}
    best = max(w.values())
    argmax = [P for P in PS if w[P] == best]
    # tie-break: highest train-window mean net return among the argmax set
    PSTAR[c] = int(max(argmax, key=lambda P: tr[tr.P == P][f'ret_{c}bp'].mean()))
    PSTAR_MEAN[c] = int(tr.groupby('P')[f'ret_{c}bp'].mean().idxmax())
    WINTAB[c] = {'wins': w, 'max': best, 'argmax': argmax}

print('\n=== train-window win matrix (ours > B&H, entry cost charged), n=35 ===')
print('P\\cost ' + ''.join(f'{str(c)+"bp":>8}' for c in COSTS))
for P in PS:
    print(f'{P:>6} ' + ''.join(f'{WINTAB[c]["wins"][P]:>8}' for c in COSTS))
print('\ncost   argmax-P set                 maxwins  P*(win rule)  P*(mean rule)  paper')
PAPER = {0: 80, 0.2: 80, 0.5: 50, 1: 40, 2: 10}
for c in COSTS:
    print(f'{str(c)+"bp":<7}{str(WINTAB[c]["argmax"]):<29}{str(WINTAB[c]["max"])+"/35":<9}'
          f'{PSTAR[c]:<14}{PSTAR_MEAN[c]:<15}{PAPER[c]}')

# ---------------- step 2: test window, frozen P* ----------------
rows = []
for i, tag in enumerate(STOCKS):
    t0 = time.time()
    X, y, pr, Yer, Yts = load_window(tag, 'test')
    up, dn = (y == 1), (y == -1)
    bh = (pr[-1] - pr[0]) / pr[0] * 100
    mo = F.dbscan_markers(Yts)
    row = {'tag': tag, 'bh': round(bh, 3)}
    for c in COSTS:
        mk = (F.dbscan_k_v2(Yer, X, k=4, minPts=30, outlier_top_pct=PSTAR[c],
                            labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
        row[f'o{c}'] = round(F.run_trading(mk & up, mk & dn, pr, c)[0], 3)
        row[f'g{c}'] = round(F.run_trading(mo & up, mo & dn, pr, c)[0], 3)
    rows.append(row)
    print(f'[test {i+1}/{len(STOCKS)}] {tag} {time.time()-t0:.0f}s', flush=True)
te = pd.DataFrame(rows)
te.to_csv(f'{OUT}/walkforward_test.csv', index=False)
json.dump({'PSTAR_win': PSTAR, 'PSTAR_mean': PSTAR_MEAN,
           'wins': {str(k): v for k, v in WINTAB.items()}},
          open(f'{OUT}/pstar.json', 'w'), indent=1)

print('\n=== TEST WINDOW (last 20 days, P* frozen on train by pilot win rule) ===')
print(f'{"cost":>5} {"P*":>4} {"ours":>7} {"orig":>7} {"B&H":>7} {"vs_BH":>7} {"vs_orig":>8}')
for c in COSTS:
    bhv = bhc(te['bh'].values, c)
    o, g = te[f'o{c}'].values, te[f'g{c}'].values
    print(f'{c:>5} {PSTAR[c]:>4} {o.mean():>7.2f} {g.mean():>7.2f} {bhv.mean():>7.2f} '
          f'{int((o > bhv).sum()):>4}/35 {int((o > g).sum()):>5}/35')
print(f'\n[saved] {OUT}/')
