"""Inspect corpus-review coverage and print small, untruncated reading sections."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def audit():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    manifest = pd.read_csv(ROOT/'review_manifest.csv',dtype={'essay_id':str})
    split = pd.read_csv(PROJECT/'splits/deberta_seed42.csv',dtype={'essay_id':str})
    train = pd.read_csv(PROJECT/'train.csv',dtype={'essay_id':str}).set_index('essay_id')
    assert manifest.essay_id.is_unique
    assert set(manifest.essay_id)==set(split.loc[split.split=='train','essay_id'])
    assert hashlib.sha256((PROJECT/'splits/deberta_seed42.csv').read_bytes()).hexdigest()==protocol['split_sha256']
    for row in manifest.itertuples():
        original=train.loc[row.essay_id]
        assert original.score==row.score
        assert hashlib.sha256(original.full_text.encode()).hexdigest()==row.text_sha256
    reviewed=manifest[manifest.batch.isin(protocol['semantic_read_completed_batches'])]
    for batch in protocol['semantic_read_completed_batches']:
        text=(ROOT/'notes'/f'{batch:04d}.md').read_text()
        for essay_id in manifest.loc[manifest.batch==batch,'essay_id']:
            assert f'| {essay_id} |' in text, f'Missing review note for {essay_id}'
    assert len(reviewed)==protocol['semantic_read_rows']
    result={'training_rows':len(manifest),'semantic_read_rows':len(reviewed),
        'remaining_rows':len(manifest)-len(reviewed),'completed_batches':protocol['semantic_read_completed_batches'],
        'next_batch':protocol['next_batch'],'training_only':True,
        'note':'Checks coverage bookkeeping and note existence; human/model reading itself is evidenced by recorded review, not proven by this script.'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=int)
    p.add_argument('--section',type=int,choices=[1,2,3],default=1)
    args=p.parse_args()
    if args.batch is None:
        print(json.dumps(audit(),indent=2))
        return
    manifest=pd.read_csv(ROOT/'review_manifest.csv',dtype={'essay_id':str})
    selected=manifest[manifest.batch==args.batch].iloc[(args.section-1)*6:args.section*6]
    train=pd.read_csv(PROJECT/'train.csv',dtype={'essay_id':str}).set_index('essay_id')
    for row in selected.itertuples():
        text=train.loc[row.essay_id,'full_text']
        assert hashlib.sha256(text.encode()).hexdigest()==row.text_sha256
        print(f'=== ID {row.essay_id} | score {row.score} | words {len(text.split())} ===\n{text}\n')


if __name__=='__main__':
    main()
