"""[Table 4] Block bootstrap for the cross-sectional mean difference.

The thirty-five per-stock returns share one 60-day window, so nominal paired
p-values overstate independence. This resamples whole trading days, the same
days for every stock at once, which preserves the cross-sectional correlation,
and rebuilds the cross-sectional mean of the total log-return difference.

Prints the observed gap, bootstrap standard error, t-like ratio and p-value
against the original detector and buy-and-hold at 0, 0.5 and 2 bp.
Run from the repository root:  python src/bootstrap_tests.py
"""
import sys,os,numpy as np,pandas as pd
sys.path.insert(0, 'src'); from sklearn.preprocessing import StandardScaler
import insample_fixP as F
rng=np.random.default_rng(0)
STOCKS=['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
 'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
 'EXR','HE','POR','TRUP','PHIN']
PSTAR={0:80,0.2:80,0.5:50,1:40,2:10}

def equity_logret(up,dn,prices,cost_bps):
    """bar-level log returns of the long/flat strategy (mirrors run_trading exactly)."""
    cr=cost_bps/10000/2; cash,pos,state=10000.0,0.0,'C'; eq=np.empty(len(prices))
    for i,p in enumerate(prices):
        if dn[i] and state=='C' and cash>0:
            cash-=cash*cr; pos=cash/p; cash=0; state='P'
        elif up[i] and state=='P' and pos>0:
            cash=pos*p; cash-=cash*cr; pos=0; state='C'
        eq[i]=cash if state=='C' else pos*p
    return np.diff(np.log(eq),prepend=np.log(10000.0))

L={}
for tag in STOCKS:
    df=pd.read_csv(f'data/{tag}_n.csv')
    cols=[c for c in df.columns if c!='Date']
    X=StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    y=np.where(df.marketClose>=df.marketOpen,1,-1); pr=df.marketClose.values.astype(float)
    up,dn=(y==1),(y==-1)
    Yer=np.load(f'cache/{tag}_ertsne_lam1.0.npy'); Yts=np.load(f'cache/{tag}_tsne.npy')
    mo=F.dbscan_markers(Yts)
    L[tag]={'up':up,'dn':dn,'pr':pr,'Yer':Yer,'X':X,'y':y,'mo':mo}
print('loaded',len(L),'stocks',flush=True)

def panel(cost,comp):
    A=[];B=[]
    for tag,d in L.items():
        mk=(F.dbscan_k_v2(d['Yer'],d['X'],k=4,minPts=30,outlier_top_pct=PSTAR[cost],
                          labels_dir=d['y'],dlp_k=10,dlp_weight=1.0)==4)
        A.append(equity_logret(mk&d['up'],mk&d['dn'],d['pr'],cost))
        if comp=='orig':
            B.append(equity_logret(d['mo']&d['up'],d['mo']&d['dn'],d['pr'],cost))
        else:  # buy and hold: long from bar0, single entry cost
            r=np.diff(np.log(d['pr']),prepend=np.log(d['pr'][0]))
            r[0]=np.log(1-cost/20000); B.append(r)
    n=min(min(len(a) for a in A),min(len(b) for b in B))
    return np.array([a[:n] for a in A]), np.array([b[:n] for b in B])

def statboot(D,mean_block=78,B=5000):
    """D: (n_stocks, T) difference panel. Statistic = cross-sectional mean of total log return."""
    n,T=D.shape; obs=D.sum(1).mean(); p=1.0/mean_block; out=np.empty(B)
    for b in range(B):
        idx=np.empty(T,dtype=np.int64); i=rng.integers(T); 
        for t in range(T):
            idx[t]=i
            i=rng.integers(T) if rng.random()<p else (i+1)%T
        out[b]=D[:,idx].sum(1).mean()
    return obs,out

print(f'\n{"cost":>5} {"vs":>5} {"obs mean dlog":>14} {"boot SE":>9} {"t-like":>7} {"boot p":>9}')
for cost in [0,0.5,2]:
    for comp in ['orig','B&H']:
        A,B_=panel(cost,comp); D=A-B_
        obs,dist=statboot(D)
        se=dist.std(ddof=1); centred=dist-dist.mean()
        pval=(np.abs(centred)>=abs(obs)).mean()
        print(f'{cost:>5} {comp:>5} {obs:>14.4f} {se:>9.4f} {obs/se:>7.2f} {max(pval,1/5000):>9.4g}'
              + ('' if pval>0 else '  (<1/5000)'))
