"""Fig: DBSCAN+K pipeline, 顶会风格 = 每个 stage 展示真实数据的中间产物 (AAP, 与 Fig1 同股票同参).
5 panels: (a) 5-min 价格序列 -> (b) ER-t-SNE embedding -> (c) DBSCAN noise
          -> (d) score 上色 + top-P% 保留 (hero) -> (e) 价格路径上的实际买卖.
风格: 一色一语义(浅灰=clustered 深灰=noise 蓝=score 绿=buy 红=sell), 无刻度小卡片面板,
panel tag 粗体, 标题加粗左对齐, 灰色小字副注, 面板间细箭头; 矢量 PDF + 600dpi PNG。
"""
import sys
import os
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import importlib.util
spec = importlib.util.spec_from_file_location('v2', os.path.join(os.path.dirname(__file__), 'dbscan_k_v2.py'))
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)

# ---------- 真实数据: AAP 全窗, 与 Fig1 一致 (P*=10, 2bp 档) ----------
TAG, P = 'AAP', 10
df = pd.read_csv(f'data/{TAG}_n.csv')
y = np.where(df.marketClose >= df.marketOpen, 1, -1)
cols = [c for c in df.columns if c != 'Date']
X = StandardScaler().fit_transform(df[cols].values.astype(np.float64))
pr = df.marketClose.values.astype(float)
Y = np.load(f'cache/{TAG}_ertsne_lam1.0.npy')

# 复现 dbscan_k_v2 内部 (与 check_aap_decomp 相同): noise -> geo/dlp -> score -> top-P%
rng = np.random.default_rng(42)
nn = NearestNeighbors(n_neighbors=30).fit(Y)
eps = np.percentile(nn.kneighbors(Y)[0][:, -1], 25)
lab = DBSCAN(eps=eps, min_samples=30).fit_predict(Y)
O = np.where(lab == -1)[0]
members = {cl: np.where(lab == cl)[0] for cl in sorted(set(lab[lab >= 0]))}
dens = {}
for cl, m in members.items():
    s = rng.choice(m, size=50, replace=False) if len(m) > 50 else m
    pw = pairwise_distances(Y[s]); dens[cl] = pw[np.triu_indices(len(s), k=1)].mean()
ncl = np.empty(len(O), int); nd = np.full(len(O), np.inf)
for cl, m in members.items():
    d = NearestNeighbors(n_neighbors=1).fit(Y[m]).kneighbors(Y[O])[0].ravel()
    c = d < nd; nd[c] = d[c]; ncl[c] = cl
geo = np.array([nd[j] / dens[ncl[j]] for j in range(len(O))])
dlp_f = 1.0 - v2._compute_local_dlp(Y, X, O, k=10)
score = (geo - geo.min()) / (geo.max() - geo.min() + 1e-10) \
      + (dlp_f - dlp_f.min()) / (dlp_f.max() - dlp_f.min() + 1e-10)
Oy = y[O]; sel = np.zeros(len(O), bool)
for d0 in [1, -1]:
    dm = Oy == d0
    sel[dm] = score[dm] >= np.percentile(score[dm], 100 - P)
buy_idx = O[sel & (Oy == -1)]   # down-marker -> buy
sell_idx = O[sel & (Oy == 1)]   # up-marker  -> sell

# 实际成交 = run_trading 的交替逻辑 (空仓遇 buy 才买, 持仓遇 sell 才卖), 与 Fig1 的 34/33 口径一致
bm = np.zeros(len(pr), bool); bm[buy_idx] = True
sm = np.zeros(len(pr), bool); sm[sell_idx] = True
state, ebuy, esell = 'C', [], []
for i in range(len(pr)):
    if bm[i] and state == 'C':
        ebuy.append(i); state = 'P'
    elif sm[i] and state == 'P':
        esell.append(i); state = 'C'

# ---------- 语义色 ----------
C_NORM, C_NOISE = '#d9d9d9', '#8a8a8a'
C_BUY, C_SELL, C_ACC, C_INK = '#2e9e5b', '#d1493e', '#3d6fb6', '#333333'
FS_T, FS_S = 8.5, 7.0

from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.80),
                         gridspec_kw=dict(wspace=0.35, left=0.005, right=0.995,
                                          top=0.86, bottom=0.22))
for a in axes:
    a.set_box_aspect(1)          # 所有面板正方形
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color('#cccccc'); s.set_linewidth(0.8)

def head(a, tag, s, hero=False):
    a.set_title(f'{tag}  {s}', loc='center', fontsize=FS_T, fontweight='bold',
                color=C_ACC if hero else C_INK, pad=3)

def foot(a, s):
    a.text(0.5, -0.32, s, transform=a.transAxes, ha='center', va='top',
           fontsize=FS_S, color='#777777')

def key(a, items):
    """frameless 迷你图例, 紧贴面板下方: items = [(marker, color, label), ...]"""
    hs = [Line2D([], [], marker=m, ls='none', markerfacecolor=c, markeredgecolor='none',
                 markersize=3.2) for m, c, _ in items]
    a.legend(hs, [l for _, _, l in items], loc='upper center',
             bbox_to_anchor=(0.5, -0.035), ncol=len(items), frameon=False,
             fontsize=FS_S, handlelength=0.6, handletextpad=0.18,
             columnspacing=0.7, borderaxespad=0, labelcolor='#555555')

# (a) 价格序列
ax = axes[0]
ax.plot(pr, color=C_INK, lw=0.45)
head(ax, 'a', 'Price series'); foot(ax, f'{TAG}, 5-min close prices, 60 days')
ax.margins(x=0.02, y=0.08)

# (b) embedding 全浅灰
ax = axes[1]
ax.scatter(Y[:, 0], Y[:, 1], s=0.4, c=C_NORM, lw=0, rasterized=True)
head(ax, 'b', 'SR-tSNE map'); foot(ax, 'one dot = one 5-min interval')

# (c) DBSCAN: clustered 浅灰 + noise 深灰
ax = axes[2]
ax.scatter(Y[lab >= 0, 0], Y[lab >= 0, 1], s=0.4, c=C_NORM, lw=0, rasterized=True)
ax.scatter(Y[O, 0], Y[O, 1], s=1.2, c=C_NOISE, lw=0, rasterized=True)
head(ax, 'c', 'DBSCAN noise'); foot(ax, 'outliers = candidates')
key(ax, [('o', C_NORM, 'clustered'), ('o', C_NOISE, 'noise')])

# (d) hero: noise 按 score 蓝色渐变, top-P% 保留成买卖色
ax = axes[3]
ax.scatter(Y[lab >= 0, 0], Y[lab >= 0, 1], s=0.4, c='#ececec', lw=0, rasterized=True)
rk = score.argsort().argsort() / max(len(score) - 1, 1)
ax.scatter(Y[O[~sel], 0], Y[O[~sel], 1], s=1.2, c=plt.cm.Blues(0.2 + 0.5 * rk[~sel]),
           lw=0, rasterized=True)
ax.scatter(Y[buy_idx, 0], Y[buy_idx, 1], s=5, c=C_BUY, edgecolors='white', lw=0.3, zorder=5)
ax.scatter(Y[sell_idx, 0], Y[sell_idx, 1], s=5, c=C_SELL, edgecolors='white', lw=0.3, zorder=5)
head(ax, 'd', 'Score & filter', hero=True)
foot(ax, 'keep top-P% by score')
key(ax, [('o', '#7fb0d8', 'score'), ('o', C_BUY, 'buy'), ('o', C_SELL, 'sell')])
for s in ax.spines.values():
    s.set_color(C_ACC); s.set_linewidth(1.2)

# (e) 价格路径 + 实际成交 (交替逻辑, 与 Fig1 口径一致)
ax = axes[4]
ax.plot(pr, color=C_INK, lw=0.45)
ax.scatter(ebuy, pr[ebuy], marker='^', s=8, c=C_BUY, edgecolors='white', lw=0.3, zorder=5)
ax.scatter(esell, pr[esell], marker='v', s=8, c=C_SELL, edgecolors='white', lw=0.3, zorder=5)
head(ax, 'e', 'Executed trades')
foot(ax, 'buy low, sell high')
key(ax, [('^', C_BUY, 'buy'), ('v', C_SELL, 'sell')])
ax.margins(x=0.02, y=0.08)

# 面板间细箭头 (fig 坐标, 取相邻 axes bbox 的中缝)
fig.canvas.draw()
for i in range(4):
    b0 = axes[i].get_position(); b1 = axes[i + 1].get_position()
    ym = (b0.y0 + b0.y1) / 2
    fig.patches.append(FancyArrowPatch((b0.x1 + 0.004, ym), (b1.x0 - 0.004, ym),
                                       transform=fig.transFigure,
                                       arrowstyle='-|>,head_width=2.2,head_length=3.8',
                                       color='#9a9a9a', lw=1.0))

fig.savefig('results/fig_pipeline.png',
            dpi=600, bbox_inches='tight')
fig.savefig('results/fig_pipeline.pdf',
            bbox_inches='tight')
print(f'[saved] fig_pipeline  noise={len(O)}  buy={len(buy_idx)}  sell={len(sell_idx)}')
