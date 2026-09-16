"""Inspect attention diversity on 18 score-balanced train essays; CPU only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from essay_bottleneck.model import BottleneckRegressor,positions
from verify_deberta_run import sha256

torch.set_num_threads(2)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',type=Path,default=Path('runs/rlt_frozen_v1'))
parser.add_argument('--output-path',type=Path,help='Optional diagnostic output path')
args=parser.parse_args()
root=Path(__file__).resolve().parent
run=args.run_dir.resolve()
config=json.loads((run/'config.json').read_text())
split=pd.read_csv(root/'splits/deberta_seed42.csv',dtype={'essay_id':str})
frame=pd.read_csv(root/'train.csv',dtype={'essay_id':str})
frame=frame[frame.essay_id.isin(split.loc[split.split=='train','essay_id'])].groupby('score',group_keys=False).head(3).reset_index(drop=True)
assert len(frame)==18
checkpoint_sha=sha256(run/'model/model.pt')
model=BottleneckRegressor.load(run/'model','cpu')
if checkpoint_sha != sha256(run/'model/model.pt'):
    raise ValueError('Checkpoint changed while loading; rerun on a stable snapshot')
tokenizer=AutoTokenizer.from_pretrained(run/'model',local_files_only=True)
rows=[]
with torch.inference_mode():
    for start in range(0,len(frame),2):
        part=frame.iloc[start:start+2]
        options={'truncation':False} if config['max_length'] is None else {'truncation':True,'max_length':config['max_length']}
        inputs=tokenizer(part.full_text.tolist(),padding=True,return_tensors='pt',**options)
        hidden=model.encoder(**inputs).last_hidden_state.float()
        mask=inputs['attention_mask'].bool()
        readout=model.readout
        memory=readout.projection(hidden)
        memory=readout.input_norm(memory+positions(hidden.shape[1],memory.shape[-1],hidden.device))
        q=readout.query_vectors(len(part))
        _,weights=readout.attention(q,memory,memory,key_padding_mask=~mask,need_weights=True,average_attn_weights=False)
        latents=readout(hidden,mask)
        for j,(_,essay) in enumerate(part.iterrows()):
            valid=mask[j]; w=weights[j,:,:,valid]
            entropy=-(w*w.clamp_min(1e-12).log()).sum(-1)/np.log(int(valid.sum()))
            normalized=F.normalize(latents[j],dim=-1)
            cosine=normalized@normalized.T
            off=~torch.eye(cosine.shape[0],dtype=torch.bool)
            rows.append({'essay_id':essay.essay_id,'score':int(essay.score),'tokens':int(valid.sum()),'mean_normalized_attention_entropy':float(entropy.mean()),'mean_latent_pair_cosine':float(cosine[off].mean()),'min_latent_pair_cosine':float(cosine[off].min())})
result={'run':str(run),'checkpoint_sha256':checkpoint_sha,'split':'train','rows':rows,'precision':'CPU FP32','mean_attention_entropy':float(np.mean([r['mean_normalized_attention_entropy'] for r in rows])),'mean_latent_pair_cosine':float(np.mean([r['mean_latent_pair_cosine'] for r in rows])),'interpretation':'Small score-balanced train diagnostic only. Entropy near 1 means attention is near uniform; cosine near 1 means latent vectors are similar. Neither alone proves redundant predictive information or a causal performance limitation.'}
output=args.output_path or root/'reports'/run.name/'latent_diagnostic.json'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
