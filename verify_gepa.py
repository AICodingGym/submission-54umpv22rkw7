"""Audit completed GEPA artifacts and recompute development metrics offline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deberta_baseline import FIXED, ROOT, integer_scores, metrics, write_json
from gepa_qwen import direct_metrics


def main():
    out=ROOT/'outputs_gepa'
    split=pd.read_csv(ROOT/'splits/deberta_seed42.csv',dtype={'essay_id':str}).set_index('essay_id')
    truth=pd.read_csv(ROOT/'train.csv',dtype={'essay_id':str}).set_index('essay_id')
    report=json.loads((out/'report.json').read_text())
    thresholds=np.array(json.loads((out/'thresholds.json').read_text())['thresholds'])
    assert thresholds.shape==(5,) and np.isfinite(thresholds).all() and (np.diff(thresholds)>0).all()
    audited=0
    diagnostics={}
    for name in ['selection','calibration']:
        p=pd.read_csv(out/f'{name}_predictions.csv',dtype={'essay_id':str})
        assert p.essay_id.is_unique
        assert set(p.essay_id)==set(split.index[split.split==name])
        np.testing.assert_array_equal(p.score,truth.loc[p.essay_id,'score'])
        probs=p[[f'p_{i}' for i in range(1,7)]].to_numpy()
        assert np.isfinite(probs).all() and (probs>=0).all()
        np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
        np.testing.assert_allclose(probs@np.arange(1,7),p.raw_prediction,atol=1e-6)
        np.testing.assert_array_equal(probs.argmax(1)+1,p.prediction)
        np.testing.assert_array_equal(integer_scores(p.raw_prediction.to_numpy(),thresholds),p.G2)
        audited+=len(p)
        diagnostics[name]={}
        for truncated in [False,True]:
            group=p[p.truncated==truncated]
            diagnostics[name]['truncated' if truncated else 'untruncated']={
                'G1':direct_metrics(group),
                'G2':metrics(group.score.to_numpy(),group.raw_prediction.to_numpy(),thresholds)}
        if name=='selection':
            for version,actual in [('G1',direct_metrics(p)),('G1_expected_fixed',metrics(p.score.to_numpy(),p.raw_prediction.to_numpy(),FIXED)),('G2',metrics(p.score.to_numpy(),p.raw_prediction.to_numpy(),thresholds))]:
                assert abs(actual['qwk']-report[version]['qwk'])<1e-12
                assert abs(actual['mae']-report[version]['mae'])<1e-12
                assert actual['confusion_matrix']==report[version]['confusion_matrix']
    base=pd.read_csv(out/'candidate_0_selection.csv',dtype={'essay_id':str})
    assert abs(direct_metrics(base)['qwk']-report['G0']['qwk'])<1e-12
    assert report['final_validation_evaluated'] is False
    for name in ['train','selection']:
        ids=pd.read_csv(out/f'search_{name}_ids.csv',dtype={'essay_id':str}).essay_id
        assert (split.loc[ids,'split']==name).all()
    # Ensure all reflection excerpts/labels originate in search training rows.
    ids=pd.read_csv(out/'search_train_ids.csv',dtype={'essay_id':str}).essay_id
    allowed={(truth.loc[i,'full_text'][:2200],int(truth.loc[i,'score'])) for i in ids}
    proposals=[json.loads(p.read_text()) for p in out.glob('proposal_*.json')]
    for proposal in proposals:
        for feedback in proposal['training_feedback']['instructions']:
            assert (feedback['essay_excerpt'],feedback['true_score']) in allowed
    write_json(out/'diagnostics.json',diagnostics)
    result={'audited_development_rows':audited,'proposal_count':len(proposals),
        'unchanged_proposals':sum(p['instructions']==p['parent'] for p in proposals),
        'reflection_training_only':True,'metrics_recomputed':True,'final_validation_evaluated':False}
    write_json(out/'verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
