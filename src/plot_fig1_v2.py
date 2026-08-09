"""Legacy method illustration (not used in the paper).
Usage: python src/plot_fig1_v2.py scan          # rank all 35 tickers
       python src/plot_fig1_v2.py fig TAG ...    # draw the named tickers
Score for a completed buy/sell pair: (1 - local percentile at the buy) x local percentile at the sell x the return of that round trip.
The local percentile is taken over a window of W = 250 bars, about three sessions.
A ticker scores as its best pair.
"""
import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import insample_fixP as F

ROOT = '.'
DATA_EXT = f'{ROOT}/data'; CACHE = 'cache'; OUTDIR = f'{ROOT}/results'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO',
          'COO','WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
COST = 2; P = 10; W = 250               # 2bp -> P* = 10


def load(tag):
    df = pd.read_csv(f'{DATA_EXT}/{tag}_n.csv')
    y = np.where(df.marketClose >= df.marketOpen, 1, -1)
    cols = [c for c in df.columns if c != 'Date']
    X = StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    prices = df.marketClose.values.astype(float)
    dates = pd.to_datetime(df.Date)
    Y_er = np.load(f'{CACHE}/{tag}_srtsne_lam1.0.npy')
    return X, y, prices, dates, Y_er


def trades(tag):
    """Run the detector and the all-in/all-out engine."""
    X, y, prices, dates, Y_er = load(tag)
    is_up, is_down = (y == 1), (y == -1)
    mk = (F.dbscan_k_v2(Y_er, X, k=4, minPts=30, outlier_top_pct=P,
                        labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
    state = 'C'; buys, sells = [], []; hold = np.zeros(len(prices), bool)
    for i in range(len(prices)):
        if is_down[i] and mk[i] and state == 'C':
            buys.append(i); state = 'P'
        elif is_up[i] and mk[i] and state == 'P':
            sells.append(i); state = 'C'
        hold[i] = (state == 'P')
    pairs = list(zip(buys, sells))       # an unclosed final buy is not paired
    return X, y, prices, dates, Y_er, mk, is_up, is_down, buys, sells, pairs, hold


def pctile(prices, i):
    """Percentile of p[i] inside its local window (0 = lowest, 1 = highest)."""
    w = prices[max(0, i - W): i + W]
    return (w < prices[i]).mean()


def score_pair(prices, i, j):
    profit = (prices[j] - prices[i]) / prices[i] * 100
    return (1 - pctile(prices, i)) * pctile(prices, j) * max(profit, 0), profit


def scan():
    rows = []
    for t in STOCKS:
        X, y, prices, dates, Y_er, mk, iu, idn, buys, sells, pairs, hold = trades(t)
        best, bi, bj, bprofit = 0, -1, -1, 0
        for i, j in pairs:
            s, pf = score_pair(prices, i, j)
            if s > best:
                best, bi, bj, bprofit = s, i, j, pf
        ret = F.run_trading(mk & iu, mk & idn, prices, COST)[0]
        bh = (prices[-1] - prices[0]) / prices[0] * 100
        rows.append({'tag': t, 'buys': len(buys), 'sells': len(sells), 'ret': ret, 'bh': bh,
                     'score': best, 'bi': bi, 'bj': bj, 'pair_profit': bprofit,
                     'buy_pct': pctile(prices, bi) if bi >= 0 else np.nan,
                     'sell_pct': pctile(prices, bj) if bj >= 0 else np.nan})
        print(f'{t:<6} buys={len(buys):2d} ret={ret:+6.1f} bh={bh:+6.1f} '
              f'best_pair profit={bprofit:+6.1f}% buy@p{rows[-1]["buy_pct"]:.02f} '
              f'sell@p{rows[-1]["sell_pct"]:.02f} score={best:6.2f}', flush=True)
    df = pd.DataFrame(rows).sort_values('score', ascending=False)
    print('\n=== ranked, few-trade tickers first ===')
    print(df[df.buys <= 8].head(8).to_string(index=False))
    print('\n=== overall top 8 ===')
    print(df.head(8).to_string(index=False))


def day_ticks(dates, every=10):
    """One tick every `every` sessions; returns (bar index, label)."""
    d = dates.dt.normalize()
    first = d.ne(d.shift()).to_numpy().nonzero()[0]   # first bar of each session
    idx = first[::every]
    return idx, [dates.iloc[i].strftime('%b %d') for i in idx]


def fig(tag):
    X, y, prices, dates, Y_er, mk, is_up, is_down, buys, sells, pairs, hold = trades(tag)
    # left panel: recover the DBSCAN clusters and noise
    nn = NearestNeighbors(n_neighbors=30).fit(Y_er)
    eps = np.percentile(nn.kneighbors(Y_er)[0][:, -1], 25)
    lab = DBSCAN(eps=eps, min_samples=30).fit_predict(Y_er)
    clustered = (lab >= 0) & ~mk; noise = (lab == -1) & ~mk
    mk_buy, mk_sell = mk & is_down, mk & is_up
    best, bi, bj = 0, -1, -1
    for i, j in pairs:
        s, _ = score_pair(prices, i, j)
        if s > best: best, bi, bj = s, i, j

    fig_, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                    gridspec_kw={'width_ratios': [1, 1.35]})
    axL.scatter(Y_er[clustered, 0], Y_er[clustered, 1], s=6, c='#cfcfcf', label='clustered (normal)')
    axL.scatter(Y_er[noise, 0], Y_er[noise, 1], s=8, c='#8a8a8a', label='unselected noise')
    axL.scatter(Y_er[mk_buy, 0], Y_er[mk_buy, 1], s=26, c='#1a9850', edgecolor='k',
                lw=0.3, label='down-marker $\\to$ buy')
    axL.scatter(Y_er[mk_sell, 0], Y_er[mk_sell, 1], s=26, c='#d73027', edgecolor='k',
                lw=0.3, label='up-marker $\\to$ sell')
    axL.set_title(f'{tag}: SR-tSNE embedding with selected markers ($P^*={P}$)')
    axL.set_xlabel('SR-tSNE 1'); axL.set_ylabel('SR-tSNE 2')
    axL.legend(fontsize=8, frameon=False, loc='best'); axL.tick_params(labelsize=8)

    # right panel: bars are evenly spaced, ticks carry real dates
    x = np.arange(len(prices))
    axR.plot(x, prices, color='#333333', lw=0.9, zorder=2)
    inpos = False; start = 0
    for i in range(len(prices)):
        if hold[i] and not inpos: inpos = True; start = i
        elif not hold[i] and inpos: axR.axvspan(start, i, color='#1a9850', alpha=0.08, zorder=1); inpos = False
    if inpos: axR.axvspan(start, len(prices) - 1, color='#1a9850', alpha=0.08, zorder=1)
    axR.scatter(buys, prices[buys], marker='^', s=55, c='#1a9850', edgecolor='k',
                lw=0.4, zorder=3, label=f'buy ({len(buys)})')
    axR.scatter(sells, prices[sells], marker='v', s=55, c='#d73027', edgecolor='k',
                lw=0.4, zorder=3, label=f'sell ({len(sells)})')
    # best buy/sell pair, circled and annotated
    if bi >= 0:
        axR.scatter([bi, bj], [prices[bi], prices[bj]], s=340, facecolor='none',
                    edgecolor='#e08214', lw=2.2, zorder=4)
        rng = prices.max() - prices.min()
        axR.margins(y=0.16)                              # leave room so the annotation does not clip
        axR.annotate('buy near local low', (bi, prices[bi]), xytext=(bi, prices[bi] - 0.13 * rng),
                     ha='center', va='top', fontsize=10, color='#1a9850', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#1a9850', lw=1.4), zorder=5)
        axR.annotate('sell near local high', (bj, prices[bj]), xytext=(bj, prices[bj] + 0.13 * rng),
                     ha='center', va='bottom', fontsize=10, color='#d73027', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#d73027', lw=1.4), zorder=5)
    ticks, labels = day_ticks(dates)
    axR.set_xticks(ticks); axR.set_xticklabels(labels, fontsize=8)
    ret_o = F.run_trading(mk & is_up, mk & is_down, prices, COST)[0]
    bh_r = (prices[-1] - prices[0]) / prices[0] * 100
    axR.set_title(f'{tag}: price path with executed trades '
                  f'(ours {ret_o:+.1f}% vs. B&H {bh_r:+.1f}% @ {COST}bp)')
    axR.set_xlabel(f'Date ({dates.iloc[0].year})'); axR.set_ylabel('Price (USD)')
    axR.legend(fontsize=9, frameon=False, loc='best'); axR.margins(x=0.01)
    fig_.suptitle('DBSCAN+K marker detection: anomalous points on the embedding '
                  'map to buy/sell actions on the price path', fontsize=12, y=1.0)
    fig_.tight_layout(rect=[0, 0, 1, 0.97])
    out = f'{OUTDIR}/method_illustration_v2_{tag}.png'
    fig_.savefig(out, dpi=160, bbox_inches='tight'); plt.close(fig_)
    print(f'[saved] {out}  (buys={len(buys)} sells={len(sells)} ours={ret_o:+.1f}% bh={bh_r:+.1f}%)')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    if mode == 'scan':
        scan()
    else:
        for t in sys.argv[2:]:
            fig(t)
