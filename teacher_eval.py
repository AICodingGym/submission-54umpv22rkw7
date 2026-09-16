"""Evaluate main-assistant-authored rubric with frozen Qwen; no reflection or final holdout access."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from deberta_baseline import ROOT, FIXED, metrics, prepare_split, write_json
from gepa_qwen import LocalScorer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=Path,help='Training example manifest; exclude from diagnostic metrics')
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory to preserve every experiment')
    prompt=args.prompt.read_text().strip()
    protocol=json.loads((ROOT/'rubric_research/protocol.json').read_text())
    splitpath=ROOT/'splits/deberta_seed42.csv'
    split_hash=hashlib.sha256(splitpath.read_bytes()).hexdigest()
    assert split_hash==protocol['split_sha256']
    frame=prepare_split(pd.read_csv(ROOT/'train.csv',dtype={'essay_id':str}),splitpath)
    manifest=pd.read_csv(ROOT/'rubric_research/review_manifest.csv',dtype={'essay_id':str})
    ids=manifest.loc[manifest.batch.isin(protocol['semantic_read_completed_batches']),'essay_id']
    train=frame[(frame.split=='train') & frame.essay_id.isin(ids)].copy()
    assert len(train)==protocol['semantic_read_rows']==len(ids)
    example_ids=[]
    if args.examples:
        example_ids=json.loads(args.examples.read_text())['ids']
        assert len(example_ids)==len(set(example_ids))
        assert set(example_ids).issubset(set(train.essay_id))
        train=train[~train.essay_id.isin(example_ids)].copy()
    selection=frame[frame.split=='selection'].copy()
    assert not set(train.essay_id)&set(selection.essay_id)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    args.output.mkdir(parents=True)
    (args.output/'prompt.txt').write_text(prompt+'\n')
    scorer=LocalScorer(args.output)
    config={'model':'Qwen/Qwen3-4B','revision':scorer.snapshot.name,
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'split_sha256':split_hash,'data_sha256':hashlib.sha256((ROOT/'train.csv').read_bytes()).hexdigest(),
        'teacher':'main assistant','reflection':False,'weights_frozen':True,
        'semantic_review_rows':protocol['semantic_read_rows'],'diagnostic_train_rows':len(train),
        'example_ids':example_ids,'full_training_rows':protocol['training_rows'],
        'thinking':False,'precision':'BF16','batch_size':2,'essay_tokens':2048,'context_tokens':4096,
        'seed':42,'calibration_used':False,'final_validation_evaluated':False}
    write_json(args.output/'config.json',config)
    report={}
    start=time.monotonic()
    for name,part in [('reviewed_train',train),('selection',selection)]:
        pred=scorer.score(part.reset_index(drop=True),prompt)
        assert pred.essay_id.tolist()==part.essay_id.tolist()
        assert np.isfinite(pred.raw_prediction).all()
        pred.to_csv(args.output/f'{name}_predictions.csv',index=False)
        report[name]={'argmax':metrics(pred.score.to_numpy(),pred.prediction.to_numpy(),FIXED),
            'expected_fixed':metrics(pred.score.to_numpy(),pred.raw_prediction.to_numpy(),FIXED),
            'truncated':int(pred.truncated.sum()),'mean_digit_mass':float(pred.score_token_mass.mean()),
            'prediction_counts':{str(k):int(v) for k,v in pred.prediction.value_counts().sort_index().items()}}
        write_json(args.output/'report.json',report)
        print(json.dumps({name:report[name]}),flush=True)
    report.update(seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  final_validation_evaluated=False,calibration_used=False)
    write_json(args.output/'report.json',report)

if __name__=='__main__':
    main()
