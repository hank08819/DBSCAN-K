"""从 grid_ablation_w.csv 生成 tab_ablation.tex (DLP 权重 w 消融, 全成本档版, 替换 7/11 的 0.5bp 固定版)。
口径: 所有 w 沿用论文冻结 P*={0:80,0.2:80,0.5:50,1:40,2:10}, 与主表 tab_vsbh 同源 (w=1 行逐股复现主表)。
单元格 = mean net return% (wins vs B&H)。"""
import numpy as np, pandas as pd

GRID = 'results/grid_ablation_w.csv'
OUT = 'results/tables/tab_ablation.tex'
COSTS = [0, 0.2, 0.5, 1, 2]
PSTAR = {0: 80, 0.2: 80, 0.5: 50, 1: 40, 2: 10}
WEIGHTS = [0.0, 0.5, 1.0, 2.0]
LABEL = {0.0: '$0$ (no DLP)', 0.5: '$0.5$', 1.0: '$1$ (default)', 2.0: '$2$'}


def bhc(bh, c):
    return ((1 - c / 20000) * (bh / 100 + 1) - 1) * 100


df = pd.read_csv(GRID)
with open(OUT, 'w') as f:
    f.write('\\begin{table}[!tbp]\n\\centering\n'
            '\\caption{DLP-weight ablation at the frozen $P^\\ast(c)$.}\n'
            '\\label{tab:ablation}\n\\resizebox{\\columnwidth}{!}{%\n'
            '\\begin{tabular}{l r r r r r}\n\\toprule\n'
            'DLP weight $w$ & $0$bp & $0.2$bp & $0.5$bp & $1$bp & $2$bp \\\\\n\\midrule\n')
    for w in WEIGHTS:
        dw = df[df.w == w]
        cells = []
        for c in COSTS:
            sel = dw[dw.P == PSTAR[c]]
            m = sel[f'ret_{c}bp'].mean()
            wins = int((sel[f'ret_{c}bp'].values > bhc(sel['buyhold'].values, c)).sum())
            s = f'{m:.1f}\\,({wins})'
            cells.append(f'\\textbf{{{s}}}' if w == 1.0 else s)
        f.write(f'{LABEL[w]} & ' + ' & '.join(cells) + ' \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}}\n\\end{table}\n')
print('[saved]', OUT)

# 正文引用数字
d1 = df[df.w == 1.0]; d0 = df[df.w == 0.0]
for c in COSTS:
    s1 = d1[d1.P == PSTAR[c]][f'ret_{c}bp'].mean()
    s0 = d0[d0.P == PSTAR[c]][f'ret_{c}bp'].mean()
    print(f'cost {c}bp: full {s1:.2f}  no-DLP {s0:.2f}  gap {s1-s0:+.2f}pp')
