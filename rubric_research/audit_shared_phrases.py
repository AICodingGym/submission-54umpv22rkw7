"""Train-only repeated-phrase diagnostic; repetition is not proof of copying."""
from collections import Counter
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
frame=pd.read_csv(ROOT/'train.csv',dtype={'essay_id':str})
split=pd.read_csv(ROOT/'splits/deberta_seed42.csv',dtype={'essay_id':str})
frame=frame.merge(split[['essay_id','split']],on='essay_id',validate='one_to_one')
frame=frame[frame.split=='train']
n=8
words=[re.findall(r"\b\w+\b",s.lower()) for s in frame.full_text]
def grams(w):
    return [hashlib.blake2b(' '.join(w[i:i+n]).encode(),digest_size=16).digest() for i in range(max(0,len(w)-n+1))]
counts=Counter()
for w in words:
    counts.update(set(grams(w)))
shared={g for g,c in counts.items() if c>=10}
rows=[]
for row,w in zip(frame.itertuples(),words):
    covered=np.zeros(len(w),dtype=bool)
    for i,g in enumerate(grams(w)):
        if g in shared:covered[i:i+n]=True
    rows.append({'essay_id':row.essay_id,'score':int(row.score),'word_count':len(w),'shared_fraction':float(covered.mean()) if len(w) else 0.})
pred=pd.DataFrame(rows)
out=ROOT/'outputs_teacher_source_audit';out.mkdir(exist_ok=True)
pred.to_csv(out/'train_shared_coverage.csv',index=False)
summary={'training_rows':len(pred),'ngram_words':n,'min_document_frequency':10,'phrase_count':len(shared),
 'median_fraction_by_score':pred.groupby('score').shared_fraction.median().to_dict(),
 'note':'All fixed train only. No labels used in phrase extraction. Repeated phrasing may be legitimate quotation, common language or source use; not a plagiarism detector or a score rule.'}
(out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
