"""Builds tab_ablation.tex from grid_ablation_w.csv: DLP-weight ablation across all cost levels.
All values of w use the frozen schedule P* = {0:80, 0.2:80, 0.5:50, 1:40, 2:10}; the w=1 row reproduces the main table.
Each cell is mean net return (%) with wins vs B&H in parentheses."""
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

# numbers quoted in the paper
d1 = df[df.w == 1.0]; d0 = df[df.w == 0.0]
for c in COSTS:
    s1 = d1[d1.P == PSTAR[c]][f'ret_{c}bp'].mean()
    s0 = d0[d0.P == PSTAR[c]][f'ret_{c}bp'].mean()
    print(f'cost {c}bp: full {s1:.2f}  no-DLP {s0:.2f}  gap {s1-s0:+.2f}pp')
