import json,re
from pathlib import Path
from collections import defaultdict
import numpy as np
def main():
 d=json.loads(Path('benchmarks/lite/fixtures/lite007i/dataset.json').read_text()); a=np.load(Path.home()/'.cache/memorilla-lite/LITE-007I/embeddings.npz',allow_pickle=False); rid={x:i for i,x in enumerate(a['record_ids'].tolist())}; qid={x:i for i,x in enumerate(a['query_ids'].tolist())}; R=a['record_embeddings']; Q=a['query_embeddings']; rec={x['evidence_id']:x for x in d['records']}; sc={x['scenario_id']:x for x in d['scenarios']}; groups=defaultdict(lambda:defaultdict(list))
 for q in d['queries']:
  s=sc[q['scenario_id']]
  if s['split']!='validation': continue
  es=s['record_ids']; M=R[[rid[e] for e in es]]; qv=Q[qid[q['query_id']]]; req={es.index(e) for e in q['required_evidence_ids'] if e in es}; qt=set(re.findall(r'[a-z0-9]+',q['query'].lower())); lex=np.array([len(qt&set(re.findall(r'[a-z0-9]+',rec[e]['content'].lower())))/max(1,len(qt|set(re.findall(r'[a-z0-9]+',rec[e]['content'].lower())))) for e in es]); cos=M@qv; hy=(lex/(lex.max() or 1)+cos/(cos.max() or 1))/2
  for name,scores in [('lexical',lex),('direct_cosine',cos),('hybrid',hy)]:
   o=np.argsort(-scores); ranks={int(x):i+1 for i,x in enumerate(o)}; groups[name][str(s['collection_size'])].append({'recall_at_1':float(bool(set(o[:1])&req)),'recall_at_3':float(bool(set(o[:3])&req)),'recall_at_5':float(bool(set(o[:5])&req)),'mrr':min((1/ranks[x] for x in req if x in ranks),default=0.0),'complete_required_top5':float(bool(req) and req.issubset(set(o[:5])))})
 out={};
 for n,by in groups.items(): out[n]={k:{m:float(np.mean([x[m] for x in v])) for m in v[0]}|{'denominator':len(v)} for k,v in by.items()}
 Path('docs/memorilla-lite/audits/LITE-007J-MATCHED-VALIDATION-BASELINES.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n'); print(out)
if __name__=='__main__': main()
