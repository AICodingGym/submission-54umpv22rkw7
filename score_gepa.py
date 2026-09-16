"""Score new essays with frozen local Qwen3-4B GEPA prompts (CUDA required)."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deberta_baseline import ROOT, integer_scores
from gepa_qwen import LocalScorer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text')
    source.add_argument('--file',type=Path)
    source.add_argument('--csv',type=Path)
    parser.add_argument('--version',choices=['G0','G1','G2'],default='G2')
    args=parser.parse_args()
    torch.set_num_threads(8)
    if args.csv:
        frame=pd.read_csv(args.csv,dtype={'essay_id':str},keep_default_na=False)
        if not {'essay_id','full_text'}.issubset(frame.columns):
            parser.error('CSV requires essay_id and full_text')
    else:
        text=args.file.read_text(encoding='utf-8') if args.file else args.text
        frame=pd.DataFrame({'essay_id':['input'],'full_text':[text]})
    if frame.empty or not frame.essay_id.is_unique or frame.essay_id.str.strip().eq('').any():
        parser.error('Provide non-empty unique essay IDs')
    if not frame.full_text.map(lambda text:isinstance(text,str) and bool(text.strip())).all():
        parser.error('Provide non-empty essay text')
    frame=frame[['essay_id','full_text']].assign(score=0)  # Placeholder unused in inference.
    directory=ROOT/'outputs_gepa'
    prompt=(directory/('prompt_G0.txt' if args.version=='G0' else 'prompt_G1.txt')).read_text().rstrip('\n')
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        scorer=LocalScorer(Path(temporary))
        config=json.loads((directory/'config.json').read_text())
        if scorer.snapshot.name != config['revision']:
            raise ValueError('Base model revision differs from the frozen GEPA run')
        # Inference progress goes to stderr so stdout is valid CSV/JSON.
        import contextlib,sys
        with contextlib.redirect_stdout(sys.stderr):
            result=scorer.score(frame,prompt)
    if args.version=='G2':
        thresholds=np.array(json.loads((directory/'thresholds.json').read_text())['thresholds'])
        if thresholds.shape!=(5,) or not np.isfinite(thresholds).all() or not (np.diff(thresholds)>0).all():
            raise ValueError('Invalid calibration thresholds')
        result['prediction']=integer_scores(result.raw_prediction.to_numpy(),thresholds)
    output=result[['essay_id','prediction']].rename(columns={'prediction':'score'})
    if args.csv:
        print(output.to_csv(index=False),end='')
    else:
        print(json.dumps({'score':int(output.score.iloc[0]),'version':args.version,'truncated':bool(result.truncated.iloc[0])}))


if __name__=='__main__':
    main()
