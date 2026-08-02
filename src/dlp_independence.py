"""[Sec 3.2.3] Independence of the two score components.

Recomputes the geometric isolation and DLP components on the DBSCAN noise pool
of every ticker and reports their Spearman rank correlation, the number quoted
in the paper. A near-zero correlation is what makes the two terms complementary
rather than two measurements of the same isolation.

Run from the repository root:  python src/dlp_independence.py
"""
import sys,os,numpy as np,pandas as pd
sys.path.insert(0, 'src'); from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import importlib.util
spec=importlib.util.spec_from_file_location('v2','src/dbscan_k_v2.py')
v2=importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
STOCKS=['MGM','UAA','BX','TKO','TSN','TPR','RHI','KFY','MCHP','PCTY','BCPC','DTM','HSIC','THO','COO',
 'WAL','ESNT','JLL','USFD','EIX','TALO','MGEE','EXP','CUBE','EME','SWX','AAP','OMC','WFRD','UTHR',
 'EXR','HE','POR','TRUP','PHIN']
corrs=[]; dlp_all=[]
for tag in STOCKS:
    df=pd.read_csv(f'data/{tag}_n.csv')
    cols=[c for c in df.columns if c!='Date']
    X=StandardScaler().fit_transform(df[cols].values.astype(np.float64))
    Y=np.load(f'cache/{tag}_ertsne_lam1.0.npy')
    nn=NearestNeighbors(n_neighbors=30).fit(Y)
    eps=np.percentile(nn.kneighbors(Y)[0][:,-1],25)
    lab=DBSCAN(eps=eps,min_samples=30).fit_predict(Y)
    O=np.where(lab==-1)[0]
    cls=sorted(set(lab[lab>=0]))
    if not len(cls) or not len(O): continue
    rng=np.random.default_rng(42); dens={}
    for cl in cls:
        m=np.where(lab==cl)[0]
        s=rng.choice(m,50,replace=False) if len(m)>50 else m
        pw=pairwise_distances(Y[s]); dens[cl]=pw[np.triu_indices(len(s),1)].mean() if len(s)>1 else 1e-10
    geo=[]
    for o in O:
        best=min(cls,key=lambda cl: pairwise_distances(Y[o:o+1],Y[lab==cl]).min())
        d=pairwise_distances(Y[o:o+1],Y[lab==best]).min()
        geo.append(d/dens[best])
    dlp=v2._compute_local_dlp(Y,X,O,k=10)
    geo=np.array(geo); dlp=np.asarray(dlp)
    if len(geo)>10:
        corrs.append(spearmanr(geo,1-dlp).statistic); dlp_all.append(dlp)
c=np.array(corrs); d=np.concatenate(dlp_all)
print(f"Spearman corr( geo , 1-DLP ) over noise candidates, per stock (n={len(c)}):")
print(f"   mean {c.mean():+.3f}   median {np.median(c):+.3f}   min {c.min():+.3f}   max {c.max():+.3f}")
print(f"   |corr| < 0.2 on {(np.abs(c)<0.2).sum()}/{len(c)} stocks")
print(f"\nDLP distribution over all {len(d)} candidates:")
for q in [1,5,25,50,75,95,99]: print(f"   p{q:<3}= {np.percentile(d,q):+.3f}")
print(f"   share with DLP > 0.5 (projection artifacts) : {(d>0.5).mean():.1%}")
