"""Two panels for AAP at 2bp with P* = 10: markers on the embedding, then the executed trades.
Computation is shared with plot_fig1_v2.py; only the layout differs.
- no suptitle or panel titles; the numbers live in the caption
- axis labelled SR-tSNE, matching the paper
- right panel keeps the circled best buy/sell pair and its annotations
- 600 dpi output
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import plot_fig1_v2 as V
import insample_fixP as F

TAG = 'AAP'
X, y, prices, dates, Y_er, mk, is_up, is_down, buys, sells, pairs, hold = V.trades(TAG)
nn = NearestNeighbors(n_neighbors=30).fit(Y_er)
eps = np.percentile(nn.kneighbors(Y_er)[0][:, -1], 25)
lab = DBSCAN(eps=eps, min_samples=30).fit_predict(Y_er)
clustered = (lab >= 0) & ~mk; noise = (lab == -1) & ~mk
mk_buy, mk_sell = mk & is_down, mk & is_up
best, bi, bj = 0, -1, -1
for i, j in pairs:
    s, _ = V.score_pair(prices, i, j)
    if s > best: best, bi, bj = s, i, j
ret_o = F.run_trading(mk & is_up, mk & is_down, prices, V.COST)[0]
bh_r = (prices[-1] - prices[0]) / prices[0] * 100

C_BUY, C_SELL, C_GOLD, C_INK = '#1a9850', '#d73027', '#e08214', '#333333'
fig_, (axL, axR) = plt.subplots(1, 2, figsize=(9.5, 3.6),
                                gridspec_kw={'width_ratios': [1, 1.35],
                                             'left': 0.055, 'right': 0.995,
                                             'top': 0.965, 'bottom': 0.13, 'wspace': 0.18})

# (a) embedding panel, full frame
axL.scatter(Y_er[clustered, 0], Y_er[clustered, 1], s=4, c='#cfcfcf', lw=0,
            rasterized=True, label='clustered (normal)')
axL.scatter(Y_er[noise, 0], Y_er[noise, 1], s=6, c='#8a8a8a', lw=0,
            rasterized=True, label='unselected noise')
axL.scatter(Y_er[mk_buy, 0], Y_er[mk_buy, 1], s=20, c=C_BUY, edgecolor='k',
            lw=0.3, label='down-marker $\\to$ buy')
axL.scatter(Y_er[mk_sell, 0], Y_er[mk_sell, 1], s=20, c=C_SELL, edgecolor='k',
            lw=0.3, label='up-marker $\\to$ sell')
axL.set_xlabel('SR-tSNE 1', fontsize=8); axL.set_ylabel('SR-tSNE 2', fontsize=8)
axL.legend(fontsize=6.5, frameon=False, loc='lower left', handletextpad=0.25,
           borderaxespad=0.3, labelspacing=0.3)
axL.tick_params(labelsize=6.5, color='#999999', length=2.5)
for sp in axL.spines.values():
    sp.set_color('#999999'); sp.set_linewidth(0.8)
axL.text(0.02, 0.985, 'a', transform=axL.transAxes, ha='left', va='top',
         fontsize=11, fontweight='bold', color=C_INK)

# (b) price path, light grid
axR.spines['top'].set_visible(False); axR.spines['right'].set_visible(False)
axR.spines['left'].set_color('#999999'); axR.spines['bottom'].set_color('#999999')
axR.grid(axis='y', color='#ededed', lw=0.6, zorder=0); axR.set_axisbelow(True)
x = np.arange(len(prices))
inpos = False; start = 0
for i in range(len(prices)):
    if hold[i] and not inpos: inpos = True; start = i
    elif not hold[i] and inpos: axR.axvspan(start, i, color=C_BUY, alpha=0.08, lw=0, zorder=1); inpos = False
if inpos: axR.axvspan(start, len(prices) - 1, color=C_BUY, alpha=0.08, lw=0, zorder=1)
axR.plot(x, prices, color=C_INK, lw=0.7, zorder=2)
axR.scatter(buys, prices[buys], marker='^', s=30, c=C_BUY, edgecolor='k',
            lw=0.35, zorder=3, label=f'buy ({len(buys)})')
axR.scatter(sells, prices[sells], marker='v', s=30, c=C_SELL, edgecolor='k',
            lw=0.35, zorder=3, label=f'sell ({len(sells)})')
if bi >= 0:
    axR.scatter([bi, bj], [prices[bi], prices[bj]], s=200, facecolor='none',
                edgecolor=C_GOLD, lw=1.8, zorder=4)
    rng = prices.max() - prices.min()
    axR.margins(y=0.16)
    axR.annotate('buy near local low', (bi, prices[bi]), xytext=(bi, prices[bi] - 0.13 * rng),
                 ha='center', va='top', fontsize=8, color=C_BUY, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_BUY, lw=1.2), zorder=5)
    axR.annotate('sell near local high', (bj, prices[bj]), xytext=(bj, prices[bj] + 0.13 * rng),
                 ha='center', va='bottom', fontsize=8, color=C_SELL, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_SELL, lw=1.2), zorder=5)
ticks, labels = V.day_ticks(dates)
axR.set_xticks(ticks); axR.set_xticklabels(labels, fontsize=6.5)
axR.tick_params(labelsize=6.5, color='#999999', length=2.5)
axR.set_xlabel(f'Date ({dates.iloc[0].year})', fontsize=8)
axR.set_ylabel('Price (USD)', fontsize=8)
axR.legend(fontsize=6.5, frameon=False, loc='lower right', handletextpad=0.25,
           borderaxespad=0.3, labelspacing=0.3)
axR.margins(x=0.01)
axR.text(0.015, 0.985, 'b', transform=axR.transAxes, ha='left', va='top',
         fontsize=11, fontweight='bold', color=C_INK)

fig_.savefig('results/fig_markers_trades.png', dpi=600, bbox_inches='tight')
print(f'[saved] fig_markers_trades  buys={len(buys)} sells={len(sells)} '
      f'ours={ret_o:+.1f}% bh={bh_r:+.1f}%')
