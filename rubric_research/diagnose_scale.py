"""Train-only ranking/scale diagnostic, never a replacement for held-out evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score,mean_absolute_error

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('runs',type=Path,nargs='+')
a=p.parse_args()
frames={str(d):pd.read_csv(d/'reviewed_train_predictions.csv',dtype={'essay_id':str}) for d in a.runs}
common=set.intersection(*(set(f.essay_id) for f in frames.values()))
out={}
for name,f in frames.items():
    f=f[f.essay_id.isin(common)].sort_values('essay_id')
    y=f.score.to_numpy();raw=f.raw_prediction.to_numpy();oof=np.zeros(len(f))
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=42).split(raw,y):
        model=IsotonicRegression(y_min=1,y_max=6,out_of_bounds='clip')
        model.fit(raw[tr],y[tr]);oof[va]=model.predict(raw[va])
    pred=np.searchsorted([1.5,2.5,3.5,4.5,5.5],oof,side='right')+1
    out[name]={'rows':len(f),'spearman':float(spearmanr(y,raw).statistic),
        'train_5fold_isotonic_qwk':float(cohen_kappa_score(y,pred,weights='quadratic',labels=list(range(1,7)))),
        'train_5fold_isotonic_mae':float(mean_absolute_error(y,pred)),
        'note':'Exploratory cross-fitted train-only scale diagnostic. Reviewed corpus is score-balanced and used for prompt development. Not independent validation; no calibration/final partition used.'}
print(json.dumps(out,indent=2))
