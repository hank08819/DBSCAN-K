"""Cost-crossover figure.
Left: mean net return vs cost for all seven methods, labelled at the line ends. Right: ours vs original at 2bp, one point per ticker.
Ours, original and B&H come from panel_35.csv; the classical baselines from baselines_perstock35.csv.
B&H and the random control are grey and distinguished by line style."""
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt

OUTDIR = 'results'
BASE_CSV = 'results/baselines_perstock35.csv'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO',
          'COO','WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
COSTS = [0, 0.2, 0.5, 1, 2]
C_OURS, C_ORIG, C_BH = '#2166ac', '#c0392b', '#9aa0a6'
C_RSI, C_FADE, INK = '#018571', '#c9c9c9', '#222222'
# shared plot style
mpl.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'axes.edgecolor': '#555', 'axes.linewidth': 0.9, 'axes.spines.top': False,
    'axes.spines.right': False, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'savefig.dpi': 200})

df = pd.read_csv(f'{OUTDIR}/panel_35.csv').set_index('tag')
o = {c: np.array([df.loc[t, f'o{c}'] for t in STOCKS]) for c in COSTS}
g = {c: np.array([df.loc[t, f'g{c}'] for t in STOCKS]) for c in COSTS}
bh0 = np.array([df.loc[t, 'bh'] for t in STOCKS])
def bhc(bh, c): return ((1 - c / 20000) * (bh / 100 + 1) - 1) * 100

x = np.array(COSTS)
om = np.array([o[c].mean() for c in COSTS])
gm = np.array([g[c].mean() for c in COSTS])
bm = np.array([bhc(bh0, c).mean() for c in COSTS])
# classical baseline means, matching tab_baseline
bl = pd.read_csv(BASE_CSV).pivot_table(index='method', columns='cost_bp', values='ret')
ma, rsi, mom, rnd = (bl.loc[m, COSTS].values for m in ['MA(5/20)', 'RSI(14)', 'Momentum', 'Random'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={'width_ratios': [1.25, 1]})
def panel_tag(ax, s):
    ax.text(-0.13, 1.06, s, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top')

# ---- left panel: mean net return vs cost ----
ax1.grid(color='#ededed', lw=0.7, zorder=0)
ax1.axhline(0, color=C_BH, lw=0.8, zorder=1)
ax1.plot(x, ma, color=C_FADE, lw=1.1, ls='-', zorder=2)
ax1.plot(x, mom, color=C_FADE, lw=1.1, ls='--', zorder=2)
ax1.plot(x, rnd, color=C_FADE, lw=1.1, ls=':', zorder=2)
ax1.plot(x, bm, '--', color=C_BH, lw=1.8, zorder=3)
ax1.plot(x, rsi, '-v', color=C_RSI, lw=1.6, ms=5, mec='white', mew=0.8, zorder=4)
ax1.plot(x, gm, '-s', color=C_ORIG, lw=2.0, ms=6, mec='white', mew=0.9, zorder=5)
ax1.plot(x, om, '-o', color=C_OURS, lw=2.6, ms=7, mec='white', mew=1, zorder=6)
# label lines at their right end; nudge apart so labels do not overlap
ends = [('DBSCAN+K (ours)', om[-1], C_OURS, 'bold'), ('t-SNE+DBSCAN (orig.)', gm[-1], C_ORIG, 'normal'),
        ('Buy & Hold', bm[-1], C_BH, 'normal'), ('RSI', rsi[-1], C_RSI, 'normal'),
        ('Random', rnd[-1], '#777777', 'normal'), ('MA', ma[-1], '#777777', 'normal'),
        ('Momentum', mom[-1], '#777777', 'normal')]
ends.sort(key=lambda e: e[1])
GAP = 3.0                                      # minimum vertical gap between labels (pp)
ys = [e[1] for e in ends]
for i in range(1, len(ys)):                    # push overlapping labels apart from the bottom up
    ys[i] = max(ys[i], ys[i - 1] + GAP)
for (name, yv, col, fw), yl in zip(ends, ys):
    if abs(yl - yv) > 0.8:                     # draw a leader line when a label has been moved
        ax1.plot([2.02, 2.06], [yv, yl], color='#cccccc', lw=0.7, clip_on=False, zorder=1)
    ax1.annotate(name, (x[-1], yv), xytext=(2.08, yl), fontsize=10, color=col,
                 fontweight=fw, va='center', annotation_clip=False)
ax1.set_xlim(-0.05, 2.05)
ax1.set_xlabel('Transaction cost (bp)'); ax1.set_ylabel('Mean net return (%)')
ax1.set_title('Ours holds up as cost rises', loc='left', fontweight='bold', fontsize=11)
panel_tag(ax1, 'a')

# ---- right panel: ours vs original at 2bp ----
gg, oo = g[2], o[2]
lo, hi = min(gg.min(), oo.min()) - 3, max(gg.max(), oo.max()) + 3
ax2.grid(color='#ededed', lw=0.7, zorder=0)
ax2.fill_between([lo, hi], [lo, hi], hi, color=C_OURS, alpha=0.06, zorder=1)   # upper triangle: ours wins
ax2.plot([lo, hi], [lo, hi], ls=(0, (5, 3)), color='#888', lw=1.2, zorder=2)
ax2.axhline(0, color=C_BH, lw=0.7, zorder=1); ax2.axvline(0, color=C_BH, lw=0.7, zorder=1)
ax2.scatter(gg, oo, s=52, c=C_OURS, alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3)
for t in ['MGM', 'RHI', 'TKO', 'AAP']:         # label the extremes and the tickers named in the text
    i = STOCKS.index(t)
    ax2.annotate(t, (gg[i], oo[i]), textcoords='offset points', xytext=(6, 3),
                 fontsize=8, fontweight='bold')
ax2.annotate('ours wins (35/35)', (0.06, 0.90), xycoords='axes fraction',
             fontsize=9.5, color=C_OURS, fontweight='bold')
ax2.annotate('original wins', (0.60, 0.06), xycoords='axes fraction', fontsize=9.5, color='#888')
ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi); ax2.set_aspect('equal')
ax2.set_xlabel('Original detector, net return @2bp (%)')
ax2.set_ylabel('DBSCAN+K (ours), net return @2bp (%)')
ax2.set_title('Every stock above the diagonal', loc='left', fontweight='bold', fontsize=11)
panel_tag(ax2, 'b')

fig.tight_layout(w_pad=2.0)
out = f'{OUTDIR}/cost_crossover_35.png'
fig.savefig(out, bbox_inches='tight'); plt.close(fig)
print('ours_mean', np.round(om, 2))
print('orig_mean', np.round(gm, 2))
print('bh_mean  ', np.round(bm, 2))
print('MA/RSI/Mom/Rnd @2bp', np.round([ma[-1], rsi[-1], mom[-1], rnd[-1]], 2))
print('@2bp above diagonal:', int((oo > gg).sum()), '/', len(oo))
print('[saved]', out)
