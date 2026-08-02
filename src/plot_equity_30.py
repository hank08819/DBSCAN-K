"""35 只 strict yahoo 赢家: 单张 0.5bp 净值曲线网格 (7 行 x 5 列 = 35 格), 占满 paper 一页。
每格 7 方法: ours/orig/B&H/RSI 彩色主线, MA/Momentum/Random 淡灰细线当背景 (小格里 7 条彩线会糊)。
baseline 信号复用 compute_baselines.py (与 Table 3/4 同源, Random 种子同 perstock 版 1000*ti+50+s)。
终点核对: ours/orig/B&H vs winners_35.csv, MA/RSI/Mom/Random vs baselines_perstock35.csv。"""
import sys
sys.path.insert(0, 'src')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import insample_fixP as F
from compute_baselines import sig_ma, sig_rsi, sig_mom, sig_random   # ⚠ import 会先跑一遍其模块级 @0.5bp 汇总 (慢 1-2 分钟, 无害)

ROOT = '.'
DATA_EXT = f'{ROOT}/data'; CACHE = 'cache'; OUTDIR = f'{ROOT}/results'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO',
          'COO','WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
LAM = 1.0; COST = 0.5; P = 50            # 0.5bp -> P*=50
C_OURS, C_ORIG, C_BH = '#2166ac', '#d6604d', '#7f7f7f'   # 与 Fig2 (plot_crossover_30) 配色一致
C_RSI, C_FADE = '#018571', '#c9c9c9'                     # RSI 主线; MA/Mom/Random 淡化
WIN = pd.read_csv(f'{OUTDIR}/winners_35.csv').set_index('tag')
BL = pd.read_csv(f'{ROOT}/results/baselines_perstock35.csv')
BL05 = BL[BL.cost_bp == COST].set_index(['ticker', 'method']).ret   # 逐股票 @0.5bp 对账用


def load(tag):
    df = pd.read_csv(f'{DATA_EXT}/{tag}_n.csv')
    y = np.where(df.marketClose >= df.marketOpen, 1, -1)
    cols = [c for c in df.columns if c != 'Date']
    X = StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    prices = df.marketClose.values.astype(float)
    Y_tsne = np.load(f'{CACHE}/{tag}_tsne.npy'); Y_er = np.load(f'{CACHE}/{tag}_ertsne_lam{LAM}.npy')
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
print(f'=== equity {COST}bp (P*={P}) 终点核对 winners + perstock baseline ===')
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
    nb, ns = int((mk_o & is_down).sum()), int((mk_o & is_up).sum())   # random 匹配 ours @0.5bp 买/卖信号数
    rr = []
    for s in range(20):
        rng = np.random.default_rng(1000 * ti + 100 * int(COST * 10) + s)   # 同 perstock 版种子
        d, u = sig_random(nb, ns, len(prices), rng)
        rr.append(equity_curve(u, d, prices, COST))
    crnd = np.mean(rr, 0)                                # 20 seeds 逐点均值曲线 (终点=perstock 均值 ret)
    curves[t] = (co, cg, cb, cma, crsi, cmo, crnd)
    eo, egv, ebh = WIN.loc[t, 'o0.5'], WIN.loc[t, 'g0.5'], WIN.loc[t, 'bh']
    ema, ersi, emo, ernd = (BL05.loc[(t, m)] for m in ['MA(5/20)', 'RSI(14)', 'Momentum', 'Random'])
    ok = (abs(co[-1]-eo) < 0.2 and abs(cg[-1]-egv) < 0.2 and abs(cb[-1]-ebh) < 0.2 and
          abs(cma[-1]-ema) < 0.2 and abs(crsi[-1]-ersi) < 0.2 and abs(cmo[-1]-emo) < 0.2 and abs(crnd[-1]-ernd) < 0.2)
    print(f'{t:<6}{co[-1]:>+8.1f}{cg[-1]:>+8.1f}{cb[-1]:>+8.1f}{cma[-1]:>+8.1f}{crsi[-1]:>+8.1f}{cmo[-1]:>+8.1f}{crnd[-1]:>+8.1f}   {"OK" if ok else "**MISMATCH**"}')

order = sorted(STOCKS, key=lambda t: curves[t][2][-1])   # 按 B&H 排
fig, axes = plt.subplots(7, 5, figsize=(15, 16.8))       # 35 只 7x5; 压矮留出 caption 空间(缩放后图高<页高)
for ax, t in zip(axes.flat, order):
    co, cg, cb, cma, crsi, cmo, crnd = curves[t]
    x = np.arange(len(co)) / len(co) * 100
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    # 弱 baseline 淡灰细线打底 (实线/虚线/点线区分), 主线 4 条彩色盖在上面
    ax.plot(x, cma, color=C_FADE, lw=0.8, ls='-', zorder=1, label='MA crossover')
    ax.plot(x, cmo, color=C_FADE, lw=0.8, ls='--', zorder=1, label='Momentum')
    ax.plot(x, crnd, color=C_FADE, lw=0.8, ls=':', zorder=1, label='Random markers')
    ax.plot(x, cb, color=C_BH, lw=1.2, zorder=2, label='Buy & Hold')
    ax.plot(x, crsi, color=C_RSI, lw=1.0, zorder=3, label='RSI')
    ax.plot(x, cg, color=C_ORIG, lw=1.2, zorder=4, label='t-SNE + DBSCAN (orig.)')
    ax.plot(x, co, color=C_OURS, lw=1.6, zorder=5, label='DBSCAN+K (ours)')
    ax.set_title(t, fontsize=12, fontweight='bold')      # 只留股票代码, 大字加粗
    ax.tick_params(labelsize=8.5); ax.margins(x=0.02)
handles, labels = axes.flat[0].get_legend_handles_labels()
ORDER = ['DBSCAN+K (ours)', 't-SNE + DBSCAN (orig.)', 'Buy & Hold', 'RSI',
         'MA crossover', 'Momentum', 'Random markers']   # 图例主线在前, 淡线在后
hl = dict(zip(labels, handles))
fig.legend([hl[k] for k in ORDER], ORDER, loc='upper center', ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.999))
fig.tight_layout(rect=[0.006, 0.006, 1, 0.966])          # 删轴标签, 子图用满边缘空间; 顶部留 2 行图例
out = f'{OUTDIR}/equity_curves_0p5bp_35.png'
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'[saved] {out}')
