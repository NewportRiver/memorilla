import json
from pathlib import Path
import numpy as np
from lite.neural.serialization import load_model
from lite.training.direct_bilinear import load as load_direct
from lite.training.numpy_gradients import score_head_forward
from lite.training.prepared_dataset import PreparedTrainingSet
from lite.training.evaluation_metrics import ranking_metrics
root=Path.home()/'.cache/memorilla-lite/LITE-007L'; art=Path.home()/'.cache/memorilla-lite/LITE-007K/prepared-independent'; ds=PreparedTrainingSet(art,allow_validation=True); d=json.loads(Path('benchmarks/lite/fixtures/lite007k-independent/dataset.json').read_text()); qs={q['query_id']:q for q in d['queries']}; sc={s['scenario_id']:s for s in d['scenarios']};
models={'associative_final':('a',load_model(root/'associative/episode-029/checkpoint')[0]),'direct_final':('d',load_direct(root/'direct/episode-029/checkpoint')[0]),'direct_initial':('d',load_direct(root/'direct-initial/checkpoint')[0])}; out={}
for name,(kind,m) in models.items():
 for rev in (False,True):
  rows=[]; ties=0
  for i in range(len(ds)):
   e=ds.get(i); q=qs[e['query_id']]; M=e['memory_embeddings'][e['valid_mask']]; ids=[x for x,v in zip(e['evidence_ids'],e['valid_mask']) if v]; req=[ids.index(x) for x in q['required_evidence_ids'] if x in ids]; qv=e['query_embedding'];
   if rev: M=M[::-1]; ids=ids[::-1]; req=[ids.index(x) for x in q['required_evidence_ids'] if x in ids]
   scores=(score_head_forward(m.forward(qv,M,return_latent=True)[1],M,m.state,np.ones(len(M),bool))[0] if kind=='a' else qv@m@M.T); ties+=int(len(scores)!=len(set(np.round(scores,7))))
   o=np.argsort(-scores,kind='mergesort'); rows.append({'ranked_ids':o.tolist(),'required_ids':req})
  out.setdefault(name,{})['reversed' if rev else 'ordinary']={**ranking_metrics(rows),'tie_rows':ties}
ds.close(); Path('docs/memorilla-lite/audits/LITE-007L-ORDERING-SENSITIVITY.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
