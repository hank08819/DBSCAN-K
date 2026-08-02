"""Figure 4 复合分析图 (1x3): a 下行保护散点 / b 换手率vs成本 / c 三方收益分布。35只, 顶刊风格。"""
import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import insample_fixP as F

OUTDIR = 'results'
DATA_EXT = 'data'; CACHE = 'cache'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
          'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
BLUE, RED, GRAY, INK = '#2166ac', '#c0392b', '#9aa0a6', '#222222'
COSTS = [0, 0.2, 0.5, 1, 2]; PSTAR = {0: 80, 0.2: 80, 0.5: 50, 1: 40, 2: 10}
mpl.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'axes.edgecolor': '#555', 'axes.linewidth': 0.9, 'axes.spines.top': False,
    'axes.spines.right': False, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'savefig.dpi': 200})
def bhc(bh, c): return ((1 - c / 20000) * (bh / 100 + 1) - 1) * 100

df = pd.read_csv(f'{OUTDIR}/winners_35.csv')
d = df[df.tag.isin(STOCKS)].set_index('tag').loc[STOCKS].reset_index()
d['ex'] = d['o0.5'] - bhc(d['bh'], 0.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
def panel_tag(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top')

# ---- a: 下行保护散点 ----
ax = axes[0]; x, y = d['bh'].values, d['ex'].values
b1, b0 = np.polyfit(x, y, 1)
ax.grid(color='#ededed', lw=0.7, zorder=0); ax.axhline(0, color=GRAY, lw=0.8, zorder=1)
xr = np.linspace(x.min()-2, x.max()+2, 50)
ax.plot(xr, b0+b1*xr, color=RED, lw=1.8, ls=(0, (5, 2)), zorder=2)
ax.annotate(f'trend ({b1:.2f})', (xr[-1], b0+b1*xr[-1]), textcoords='offset points',
            xytext=(-2, 10), ha='right', fontsize=9.5, color=RED)   # 直标在趋势线末端, 代替图例框
ax.scatter(x, y, s=52, c=BLUE, alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3)
for t in ['MGM', 'MCHP', 'TRUP', 'WFRD']:
    r = d[d.tag == t].iloc[0]
    ax.annotate(t, (r['bh'], r['ex']), textcoords='offset points', xytext=(5, 4), fontsize=9.5, fontweight='bold')
ax.set_xlabel('Buy-and-hold return (%)'); ax.set_ylabel('Excess return, ours $-$ B&H (pp)')
ax.set_title('Edge largest on weak stocks', loc='left', fontweight='bold', fontsize=12)
panel_tag(ax, 'a')

# ---- b: 换手率 vs 成本 ----
ax = axes[1]
def load(t):
    dd = pd.read_csv(f'{DATA_EXT}/{t}_n.csv'); yy = np.where(dd.marketClose >= dd.marketOpen, 1, -1)
    cols = [c for c in dd.columns if c != 'Date']
    X = StandardScaler().fit_transform(dd[cols].values.astype(np.float64))
    return X, yy, dd.marketClose.values.astype(float), np.load(f'{CACHE}/{t}_tsne.npy'), np.load(f'{CACHE}/{t}_ertsne_lam1.0.npy')
ours_tr = {c: [] for c in COSTS}; orig_tr = []
for t in STOCKS:
    X, yy, pr, Yt, Ye = load(t); up, dn = (yy == 1), (yy == -1)
    mg = F.dbscan_markers(Yt); orig_tr.append(F.run_trading(mg & up, mg & dn, pr, 0)[1])
    for c in COSTS:
        mo = (F.dbscan_k_v2(Ye, X, k=4, minPts=30, outlier_top_pct=PSTAR[c], labels_dir=yy, dlp_k=10, dlp_weight=1.0) == 4)
        ours_tr[c].append(F.run_trading(mo & up, mo & dn, pr, c)[1])
om = np.array([np.mean(ours_tr[c]) for c in COSTS]); gm = float(np.mean(orig_tr)); xc = np.array(COSTS)
ax.grid(axis='y', color='#ededed', lw=0.7, zorder=0)
ax.fill_between(xc, om, gm, color=BLUE, alpha=0.08, zorder=1)
ax.hlines(gm, xc.min(), xc.max(), color=RED, lw=2.2, zorder=2)
ax.plot(xc, om, '-o', color=BLUE, lw=2.2, ms=8, mec='white', mew=1, zorder=3)
# 直标代替图例框: 红线上方 + 蓝线末端
ax.annotate('original (no filter)', (1.0, gm), textcoords='offset points', xytext=(0, 6),
            ha='center', fontsize=8.5, color=RED)
ax.annotate('DBSCAN+K (ours)', (xc[3], om[3]), textcoords='offset points', xytext=(0, -20),
            ha='center', fontsize=8.5, color=BLUE, fontweight='bold')   # 曲线下方空白区, 避开 P* 标注
for xi, yi, c in zip(xc, om, COSTS):
    ax.annotate(f'$P^*${PSTAR[c]}', (xi, yi), textcoords='offset points', xytext=(0, 9), fontsize=8.5, ha='center', color=BLUE)
ax.annotate('trades saved', (1.2, (om[2]+gm)/2), fontsize=9, ha='center', va='center', color='#555')
ax.set_xlabel('Transaction cost (bp)'); ax.set_ylabel('Round-trip trades / stock')
ax.set_title('Filter trades less as cost rises', loc='left', fontweight='bold', fontsize=12)
ax.set_xlim(-0.06, 2.06); ax.set_ylim(0, gm*1.14)
panel_tag(ax, 'b')

# ---- c: 三方收益分布 (0.5bp) ----
ax = axes[2]
bb = bhc(d['bh'].values, 0.5); gg = d['g0.5'].values; oo = d['o0.5'].values
groups = [bb, gg, oo]; cols = [GRAY, RED, BLUE]; labs = ['B&H', 'orig.', 'ours']
ax.grid(axis='y', color='#ededed', lw=0.7, zorder=0); ax.axhline(0, color=GRAY, lw=0.8, zorder=1)
bp = ax.boxplot(groups, positions=[1, 2, 3], widths=0.55, patch_artist=True, showfliers=False,
                medianprops=dict(color='white', lw=1.6), whiskerprops=dict(color='#888'), capprops=dict(color='#888'))
for patch, cc in zip(bp['boxes'], cols):
    patch.set_facecolor(cc); patch.set_alpha(0.55); patch.set_edgecolor(cc)
rng = np.random.default_rng(0)
for i, (grp, cc) in enumerate(zip(groups, cols), 1):
    jx = i + (rng.random(len(grp)) - 0.5) * 0.28
    ax.scatter(jx, grp, s=16, c=cc, alpha=0.75, edgecolors='white', linewidths=0.4, zorder=3)
for i, grp in enumerate(groups, 1):
    ax.annotate(f'{np.mean(grp):+.1f}', (i, np.mean(grp)), textcoords='offset points', xytext=(16, -3), fontsize=8.5, fontweight='bold')
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(labs)
ax.set_ylabel('Net return @0.5bp (%)')
ax.set_title('Ours shifts the whole distribution up', loc='left', fontweight='bold', fontsize=12)
panel_tag(ax, 'c')

fig.tight_layout(w_pad=2.4)
out = f'{OUTDIR}/fig_analysis_3panel.png'
fig.savefig(out, bbox_inches='tight'); plt.close(fig)
print(f'means @0.5bp: B&H {bb.mean():+.1f}  orig {gg.mean():+.1f}  ours {oo.mean():+.1f}')
print('[saved]', out)
