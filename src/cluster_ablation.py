"""[Table 8] Candidate generator ablation.

Replaces the DBSCAN stage with HDBSCAN and OPTICS at library defaults, matched
to minPts=30 where an equivalent knob exists. The scoring layer (geometric
isolation + DLP), the top-P% filter, the P*(c) schedule and the trading engine
are held fixed, so the only thing that varies is which bars become candidates.

Writes results/cluster_ablation.csv.
Run from the repository root:  python src/cluster_ablation.py
"""
import sys,os,numpy as np,pandas as pd,importlib.util
sys.path.insert(0, 'src'); from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from sklearn.metrics import pairwise_distances
import insample_fixP as F
spec=importlib.util.spec_from_file_location('v2','src/dbscan_k_v2.py')
v2=importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)

STOCKS=['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
 'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
 'EXR','HE','POR','TRUP','PHIN']
COSTS=[0,0.2,0.5,1,2]; PSTAR={0:80,0.2:80,0.5:50,1:40,2:10}; MINPTS=30

def candidates(name,Y):
    if name=='DBSCAN':
        eps=np.percentile(NearestNeighbors(n_neighbors=MINPTS).fit(Y).kneighbors(Y)[0][:,-1],25)
        return DBSCAN(eps=eps,min_samples=MINPTS).fit_predict(Y)
    if name=='HDBSCAN': return HDBSCAN(min_cluster_size=MINPTS).fit_predict(Y)
    if name=='OPTICS':  return OPTICS(min_samples=MINPTS).fit_predict(Y)

def score_and_filter(lab,Y,X,ydir,P):
    """verbatim port of dbscan_k_v2 Phase 2, given any (clusters, noise) labelling"""
    O=np.where(lab==-1)[0]; cls=sorted(set(lab[lab>=0]))
    if len(O)==0 or len(cls)==0: return np.zeros(len(Y),bool)
    rng=np.random.default_rng(42); dens={}
    for cl in cls:
        m=np.where(lab==cl)[0]
        s=rng.choice(m,50,replace=False) if len(m)>50 else m
        if len(s)<=1: dens[cl]=1e-10; continue
        pw=pairwise_distances(Y[s]); dens[cl]=pw[np.triu_indices(len(s),1)].mean()
    nc=np.empty(len(O),int); nd=np.full(len(O),np.inf)
    for cl in cls:
        d=NearestNeighbors(n_neighbors=1).fit(Y[lab==cl]).kneighbors(Y[O])[0].ravel()
        c=d<nd; nd[c]=d[c]; nc[c]=cl
    geo=np.array([nd[j]/dens[nc[j]] for j in range(len(O))])
    dlp=1.0-np.asarray(v2._compute_local_dlp(Y,X,O,k=10))
    gn=(geo-geo.min())/(geo.max()-geo.min()+1e-10)
    dn=(dlp-dlp.min())/(dlp.max()-dlp.min()+1e-10)
    sc=gn+1.0*dn
    keep=np.zeros(len(O),bool); ol=ydir[O]
    for d_ in (1,-1):
        m=(ol==d_)
        if m.sum(): keep[m]=sc[m]>=np.percentile(sc[m],100-P)
    out=np.zeros(len(Y),bool); out[O[keep]]=True; return out

METH=['DBSCAN','HDBSCAN','OPTICS']
rows=[]
for i,tag in enumerate(STOCKS):
    df=pd.read_csv(f'data/{tag}_n.csv')
    cols=[c for c in df.columns if c!='Date']
    X=StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    Y=np.load(f'cache/{tag}_ertsne_lam1.0.npy')
    y=np.where(df.marketClose>=df.marketOpen,1,-1); pr=df.marketClose.values.astype(float)
    up,dn=(y==1),(y==-1)
    for m in METH:
        lab=candidates(m,Y); nnoise=int((lab==-1).sum()); ncl=len(set(lab[lab>=0]))
        r={'tag':tag,'method':m,'noise':nnoise,'nclust':ncl}
        for c in COSTS:
            mk=score_and_filter(lab,Y,X,y,PSTAR[c])
            r[f'ret{c}']=F.run_trading(mk&up,mk&dn,pr,c)[0]
        rows.append(r)
    print(f'[{i+1}/35] {tag}',flush=True)
d=pd.DataFrame(rows); d.to_csv('results/cluster_ablation.csv',index=False)
bh=[]
for tag in STOCKS:
    pr=pd.read_csv(f'data/{tag}_n.csv').marketClose.values.astype(float)
    bh.append((pr[-1]-pr[0])/pr[0]*100)
bh=np.array(bh)
print(f'\n{"method":<9}{"cands":>7}{"clust":>6}'+''.join(f'{str(c)+"bp":>9}' for c in COSTS)+f'{"wins@2bp":>10}')
for m in METH:
    s=d[d.method==m]
    row=f'{m:<9}{s.noise.mean():>7.0f}{s.nclust.mean():>6.1f}'
    for c in COSTS: row+=f'{s[f"ret{c}"].mean():>9.2f}'
    w=int((s['ret2'].values>((1-2/20000)*(bh/100+1)-1)*100).sum())
    print(row+f'{w:>7}/35')
