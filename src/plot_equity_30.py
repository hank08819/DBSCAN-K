"""Equity-curve grid at 0.5bp, 7 x 5 panels, one per ticker.
Ours, original, B&H and RSI are drawn in colour; MA, momentum and the random control in light grey.
Baseline signals come from compute_baselines.py, with the same seeds as the per-ticker run.
End points are cross-checked against panel_35.csv and baselines_perstock35.csv."""
import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import insample_fixP as F
from compute_baselines import sig_ma, sig_rsi, sig_mom, sig_random   # importing this runs its module-level 0.5bp summary first (1-2 minutes)

ROOT = '.'
DATA_EXT = f'{ROOT}/data'; CACHE = 'cache'; OUTDIR = f'{ROOT}/results'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO',
          'COO','WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
LAM = 1.0; COST = 0.5; P = 50            # 0.5bp -> P*=50
C_OURS, C_ORIG, C_BH = '#2166ac', '#d6604d', '#7f7f7f'   # same colours as the cost-crossover figure
C_RSI, C_FADE = '#018571', '#c9c9c9'                     # RSI in colour; MA, momentum and random faded
PANEL = pd.read_csv(f'{OUTDIR}/panel_35.csv').set_index('tag')
BL = pd.read_csv(f'{ROOT}/results/baselines_perstock35.csv')
BL05 = BL[BL.cost_bp == COST].set_index(['ticker', 'method']).ret   # per-ticker cross-check at 0.5bp


def load(tag):
    df = pd.read_csv(f'{DATA_EXT}/{tag}_n.csv')
    y = np.where(df.marketClose >= df.marketOpen, 1, -1)
    cols = [c for c in df.columns if c != 'Date']
    X = StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    prices = df.marketClose.values.astype(float)
    Y_tsne = np.load(f'{CACHE}/{tag}_tsne.npy'); Y_er = np.load(f'{CACHE}/{tag}_srtsne_lam{LAM}.npy')
    return X, y, prices, Y_tsne, Y_er


def equity_curve(up_mask, down_mask, prices, cost_bps):
    cost_rate = cost_bps / 10000 / 2
    cash, pos, state = 10000.0, 0.0, 'C'; eq = np.empty(len(prices))
    for i in range(len(prices)):
        p = prices[i]
        if down_mask[i] and state == 'C' and cash > 0:
            cash -= cash * cost_rate; pos = cash / p; cash = 0; state = 'P'
        elif up_mask[i] and state == 'P' and pos > 0:
            cash = pos * p; cash -= cash * cost_rate; pos = 0; state = 'C'
        eq[i] = cash if state == 'C' else pos * p
    return (eq - 10000) / 100


def bh_curve(prices): return (10000.0 / prices[0] * prices - 10000) / 100


curves = {}
print(f'=== equity at {COST}bp (P*={P}): end-point cross-check ===')
print(f'{"stock":<6}{"ours":>8}{"orig":>8}{"B&H":>8}{"MA":>8}{"RSI":>8}{"Mom":>8}{"Rnd":>8}   ok')
for ti, t in enumerate(STOCKS):
    X, y, prices, Y_tsne, Y_er = load(t)
    is_up, is_down = (y == 1), (y == -1)
    mk_o = (F.dbscan_k_v2(Y_er, X, k=4, minPts=30, outlier_top_pct=P,
                          labels_dir=y, dlp_k=10, dlp_weight=1.0) == 4)
    mk_g = F.dbscan_markers(Y_tsne)
    co = equity_curve(mk_o & is_up, mk_o & is_down, prices, COST)
    cg = equity_curve(mk_g & is_up, mk_g & is_down, prices, COST)
    cb = bh_curve(prices)
    d, u = sig_ma(prices);  cma = equity_curve(u, d, prices, COST)
    d, u = sig_rsi(prices); crsi = equity_curve(u, d, prices, COST)
    d, u = sig_mom(prices); cmo = equity_curve(u, d, prices, COST)
    nb, ns = int((mk_o & is_down).sum()), int((mk_o & is_up).sum())   # random control matches our buy/sell counts
    rr = []
    for s in range(20):
        rng = np.random.default_rng(1000 * ti + 100 * int(COST * 10) + s)   # same seed scheme as the per-ticker run
        d, u = sig_random(nb, ns, len(prices), rng)
        rr.append(equity_curve(u, d, prices, COST))
    crnd = np.mean(rr, 0)                                # pointwise mean over 20 seeds
    curves[t] = (co, cg, cb, cma, crsi, cmo, crnd)
    eo, egv, ebh = PANEL.loc[t, 'o0.5'], PANEL.loc[t, 'g0.5'], PANEL.loc[t, 'bh']
    ema, ersi, emo, ernd = (BL05.loc[(t, m)] for m in ['MA(5/20)', 'RSI(14)', 'Momentum', 'Random'])
    ok = (abs(co[-1]-eo) < 0.2 and abs(cg[-1]-egv) < 0.2 and abs(cb[-1]-ebh) < 0.2 and
          abs(cma[-1]-ema) < 0.2 and abs(crsi[-1]-ersi) < 0.2 and abs(cmo[-1]-emo) < 0.2 and abs(crnd[-1]-ernd) < 0.2)
    print(f'{t:<6}{co[-1]:>+8.1f}{cg[-1]:>+8.1f}{cb[-1]:>+8.1f}{cma[-1]:>+8.1f}{crsi[-1]:>+8.1f}{cmo[-1]:>+8.1f}{crnd[-1]:>+8.1f}   {"OK" if ok else "**MISMATCH**"}')

order = sorted(STOCKS, key=lambda t: curves[t][2][-1])   # ordered by buy-and-hold return
fig, axes = plt.subplots(7, 5, figsize=(15, 16.8))       # 7 x 5 grid for 35 tickers
for ax, t in zip(axes.flat, order):
    co, cg, cb, cma, crsi, cmo, crnd = curves[t]
    x = np.arange(len(co)) / len(co) * 100
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    # faded baselines underneath, the four coloured lines on top
    ax.plot(x, cma, color=C_FADE, lw=0.8, ls='-', zorder=1, label='MA crossover')
    ax.plot(x, cmo, color=C_FADE, lw=0.8, ls='--', zorder=1, label='Momentum')
    ax.plot(x, crnd, color=C_FADE, lw=0.8, ls=':', zorder=1, label='Random markers')
    ax.plot(x, cb, color=C_BH, lw=1.2, zorder=2, label='Buy & Hold')
    ax.plot(x, crsi, color=C_RSI, lw=1.0, zorder=3, label='RSI')
    ax.plot(x, cg, color=C_ORIG, lw=1.2, zorder=4, label='t-SNE + DBSCAN (orig.)')
    ax.plot(x, co, color=C_OURS, lw=1.6, zorder=5, label='DBSCAN+K (ours)')
    ax.set_title(t, fontsize=12, fontweight='bold')      # ticker only, bold
    ax.tick_params(labelsize=8.5); ax.margins(x=0.02)
handles, labels = axes.flat[0].get_legend_handles_labels()
ORDER = ['DBSCAN+K (ours)', 't-SNE + DBSCAN (orig.)', 'Buy & Hold', 'RSI',
         'MA crossover', 'Momentum', 'Random markers']   # coloured lines first in the legend, faded ones after
hl = dict(zip(labels, handles))
fig.legend([hl[k] for k in ORDER], ORDER, loc='upper center', ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.999))
fig.tight_layout(rect=[0.006, 0.006, 1, 0.966])          # two legend rows at the top
out = f'{OUTDIR}/equity_curves_0p5bp_35.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'[saved] {out}')
