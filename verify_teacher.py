"""Independently verify saved teacher-prompt evaluation artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from deberta_baseline import ROOT,FIXED,metrics


def verify(directory):
    config=json.loads((directory/'config.json').read_text())
    report=json.loads((directory/'report.json').read_text())
    assert 'seconds' in report, 'Run not complete'
    assert not config['final_validation_evaluated'] and not config['calibration_used']
    prompt=(directory/'prompt.txt').read_text().strip()
    assert hashlib.sha256(prompt.encode()).hexdigest()==config['prompt_sha256']
    assert hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest()==config['data_sha256']
    splitpath=ROOT/'splits/deberta_seed42.csv'
    assert hashlib.sha256(splitpath.read_bytes()).hexdigest()==config['split_sha256']
    splits=pd.read_csv(splitpath,dtype={'essay_id':str})
    truth=pd.read_csv(ROOT/'train.csv',dtype={'essay_id':str}).set_index('essay_id').score
    frames={}
    for name,split in [('selection','selection'),('reviewed_train','train')]:
        pred=pd.read_csv(directory/f'{name}_predictions.csv',dtype={'essay_id':str})
        assert pred.essay_id.is_unique
        allowed=set(splits.loc[splits.split==split,'essay_id'])
        assert set(pred.essay_id).issubset(allowed)
        if split=='selection':
            assert set(pred.essay_id)==allowed
        else:
            assert len(pred)==config.get('diagnostic_train_rows',config['semantic_review_rows'])
        assert not set(pred.essay_id)&set(config.get('example_ids',[]))
        assert np.array_equal(pred.score,truth.loc[pred.essay_id].to_numpy())
        probs=pred[[f'p_{i}' for i in range(1,7)]].to_numpy()
        assert np.isfinite(probs).all() and (probs>=0).all()
        assert np.allclose(probs.sum(axis=1),1,atol=1e-6)
        assert np.array_equal(pred.prediction,probs.argmax(axis=1)+1)
        assert np.allclose(pred.raw_prediction,probs@np.arange(1,7),atol=1e-6)
        for method,col in [('argmax','prediction'),('expected_fixed','raw_prediction')]:
            actual=metrics(pred.score.to_numpy(),pred[col].to_numpy(),FIXED)
            saved=report[name][method]
            assert actual['rows']==saved['rows']
            assert actual['confusion_matrix']==saved['confusion_matrix']
            assert np.isclose(actual['qwk'],saved['qwk'])
            assert np.isclose(actual['mae'],saved['mae'])
        frames[name]=pred
    return frames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+')
    args=p.parse_args()
    runs={str(path):verify(path) for path in args.runs}
    common=set.intersection(*(set(r['reviewed_train'].essay_id) for r in runs.values()))
    result={}
    for name,r in runs.items():
        pred=r['reviewed_train'];sub=pred[pred.essay_id.isin(common)]
        result[name]={'verified':True,'selection':metrics(r['selection'].score.to_numpy(),r['selection'].prediction.to_numpy(),FIXED),
                      'common_train':metrics(sub.score.to_numpy(),sub.prediction.to_numpy(),FIXED)}
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
