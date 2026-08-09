"""Builds tab_vsbh.tex, tab_detailed.tex and tab_tickers.tex from panel_35.csv,
and prints the summary numbers quoted in the abstract and body (o = DBSCAN+K on SR-tSNE at P*, g = t-SNE + plain DBSCAN, bh = buy-and-hold)."""
import numpy as np, pandas as pd

PANEL = 'results/panel_35.csv'
OUTDIR = 'results/tables'
STOCKS = ['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO',
          'COO','WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
          'EXR','HE','POR','TRUP','PHIN']
N = len(STOCKS)
COSTS = [0, 0.2, 0.5, 1, 2]
PSTAR = {0: 80, 0.2: 80, 0.5: 50, 1: 40, 2: 10}
# company names and exchanges for the appendix table (from yfinance)
INFO = {
 'MGM':('MGM Resorts International','NYSE'),'UAA':('Under Armour, Inc.','NYSE'),
 'BX':('Blackstone Inc.','NYSE'),'TKO':('TKO Group Holdings, Inc.','NYSE'),
 'TSN':('Tyson Foods, Inc.','NYSE'),'TPR':('Tapestry, Inc.','NYSE'),
 'RHI':('Robert Half Inc.','NYSE'),'KFY':('Korn Ferry','NYSE'),
 'MCHP':('Microchip Technology Inc.','NASDAQ'),'PCTY':('Paylocity Holding Corp.','NASDAQ'),
 'BCPC':('Balchem Corporation','NASDAQ'),'DTM':('DT Midstream, Inc.','NYSE'),
 'HSIC':('Henry Schein, Inc.','NASDAQ'),'THO':('Thor Industries, Inc.','NYSE'),
 'COO':('The Cooper Companies, Inc.','NASDAQ'),'WAL':('Western Alliance Bancorp.','NYSE'),
 'ESNT':('Essent Group Ltd.','NYSE'),'JLL':('Jones Lang LaSalle Inc.','NYSE'),
 'USFD':('US Foods Holding Corp.','NYSE'),'EIX':('Edison International','NYSE'),
 'TALO':('Talos Energy, Inc.','NYSE'),'MGEE':('MGE Energy, Inc.','NASDAQ'),
 'EXP':('Eagle Materials Inc.','NYSE'),'CUBE':('CubeSmart','NYSE'),
 'EME':('EMCOR Group, Inc.','NYSE'),'SWX':('Southwest Gas Holdings, Inc.','NYSE'),
 'AAP':('Advance Auto Parts, Inc.','NYSE'),'OMC':('Omnicom Group Inc.','NYSE'),
 'WFRD':('Weatherford International plc','NASDAQ'),'UTHR':('United Therapeutics Corp.','NASDAQ'),
 'EXR':('Extra Space Storage Inc.','NYSE'),'HE':('Hawaiian Electric Industries, Inc.','NYSE'),
 'POR':('Portland General Electric Co.','NYSE'),'TRUP':('Trupanion, Inc.','NASDAQ'),
 'PHIN':('PHINIA Inc.','NYSE'),
}

df = pd.read_csv(PANEL).set_index('tag')
o = {c: np.array([df.loc[t, f'o{c}'] for t in STOCKS]) for c in COSTS}
g = {c: np.array([df.loc[t, f'g{c}'] for t in STOCKS]) for c in COSTS}
bh0 = np.array([df.loc[t, 'bh'] for t in STOCKS])
def bhc(bh, c): return ((1 - c / 20000) * (bh / 100 + 1) - 1) * 100
bh = {c: bhc(bh0, c) for c in COSTS}

# ---------- Table 1: vs buy&hold ----------
with open(f'{OUTDIR}/tab_vsbh.tex', 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{Performance against buy-and-hold under realistic '
            'costs across %d equities. $P^\\ast$ is the global cost-to-filter rule.}\n\\label{tab:vsbh}\n' % N)
    f.write('\\begin{tabular}{l r c r r}\n\\toprule\nCost & $P^\\ast$ & Wins & Ours\\% & B\\&H\\% \\\\\n\\midrule\n')
    r1 = []
    for c in COSTS:
        w = int(np.sum(o[c] > bh[c])); om = o[c].mean(); bm = bh[c].mean()
        r1.append((c, w, om, bm))
        f.write(f'{c}bp & {PSTAR[c]} & {w}/{N} & {om:.2f} & {bm:.2f} \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

# ---------- Table 2: per-stock detailed ----------
order = sorted(range(N), key=lambda i: bh0[i])
def cell(v, bold=False):
    s = f'{v:+.1f}'; return f'\\textbf{{{s}}}' if bold else s
nc = len(COSTS)
cost_hdr = ' & '.join(f'{c}bp' for c in COSTS)
col_spec = 'l r ' + 'r' * nc + ' ' + 'r' * nc
with open(f'{OUTDIR}/tab_detailed.tex', 'w') as f:
    wins = {c: int(np.sum(o[c] > g[c])) for c in COSTS}
    win_str = '/'.join(str(wins[c]) for c in COSTS)
    p_str = '/'.join(str(PSTAR[c]) for c in COSTS)
    f.write('\\begin{table*}[p]\n\\centering\n'
            '\\caption{Per-stock net return (\\%) per cost; B\\&H $=$ buy-and-hold, orig.\\ $=$ original detector.}\n'
            '\\label{tab:detailed}\n')
    f.write('\\resizebox{0.99\\textwidth}{!}{%%\n\\begin{tabular}{%s}\n\\toprule\n' % col_spec)
    f.write(' & & \\multicolumn{%d}{c}{DBSCAN+K (ours)} & \\multicolumn{%d}{c}{t-SNE + DBSCAN (orig.)} \\\\\n' % (nc, nc))
    f.write('\\cmidrule(lr){3-%d}\\cmidrule(lr){%d-%d}\n' % (2 + nc, 3 + nc, 2 + 2 * nc))
    f.write('Stock & B\\&H & %s & %s \\\\\n\\midrule\n' % (cost_hdr, cost_hdr))
    for i in order:
        t = STOCKS[i]
        oc = ' & '.join(cell(o[c][i], o[c][i] > g[c][i]) for c in COSTS)
        gc = ' & '.join(cell(g[c][i]) for c in COSTS)
        f.write(f'{t} & {bh0[i]:+.1f} & {oc} & {gc} \\\\\n')
    f.write('\\midrule\n')
    f.write('Mean & %+.1f & ' % bh0.mean() + ' & '.join(f'{o[c].mean():+.1f}' for c in COSTS) + ' & '
            + ' & '.join(f'{g[c].mean():+.1f}' for c in COSTS) + ' \\\\\n')
    f.write('\\bottomrule\n\\end{tabular}}\n\\end{table*}\n')

# ---------- appendix ticker table (two columns) ----------
FAME = {'MGM':3,'UAA':3,'BX':3,'TKO':3,'TSN':3,'TPR':3}  # prominence weight used only for ordering the appendix table
order_f = sorted(range(N), key=lambda i: (-FAME.get(STOCKS[i], 2), -o[0][i]))
with open(f'{OUTDIR}/tab_tickers.tex', 'w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{The %d U.S.\\ equities (5-min 2026 bars, Yahoo).}\n'
            '\\label{tab:tickers}\n\\resizebox{\\columnwidth}{!}{%%\n'
            '\\begin{tabular}{l l l}\n\\toprule\nTicker & Company & Exchange \\\\\n\\midrule\n' % N)
    for i in order_f:
        t = STOCKS[i]; name, exch = INFO[t]
        f.write(f'{t} & {name} & {exch} \\\\\n'.replace('&', '\\&', 0))
    f.write('\\bottomrule\n\\end{tabular}}\n\\end{table}\n')

# ---------- numbers quoted in the paper ----------
print('=== summary numbers (N=%d) ===' % N)
print('vs B&H  cost | wins | ours_mean | bh_mean')
for c in COSTS:
    print(f'  {c}bp | {int(np.sum(o[c]>bh[c]))}/{N} | {o[c].mean():+.2f} | {bh[c].mean():+.2f}')
print('vs ORIG cost | wins | ours_mean | orig_mean | Δ')
for c in COSTS:
    print(f'  {c}bp | {int(np.sum(o[c]>g[c]))}/{N} | {o[c].mean():+.2f} | {g[c].mean():+.2f} | {o[c].mean()-g[c].mean():+.2f}')
d = [o[c].mean()-g[c].mean() for c in COSTS]
print(f'\nΔ range: {min(d):+.1f} .. {max(d):+.1f} (peak @ {COSTS[int(np.argmax(d))]}bp)')
print(f'ours_mean range: {min(o[c].mean() for c in COSTS):+.1f} .. {max(o[c].mean() for c in COSTS):+.1f}')
print(f'BH mean @0bp: {bh[0].mean():+.2f}')
print('[saved] tab_vsbh.tex + tab_detailed.tex + tab_tickers.tex')
