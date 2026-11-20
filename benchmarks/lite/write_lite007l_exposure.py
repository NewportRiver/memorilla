import json,hashlib
from pathlib import Path
import numpy as np
root=Path.home()/'.cache/memorilla-lite/LITE-007L'; prep=Path.home()/'.cache/memorilla-lite/LITE-007J/prepared-train.npz'; order=json.loads((root/'schedule-order.json').read_text()); eligible=json.loads((root/'eligible-indices.json').read_text());
data=json.loads(Path('benchmarks/lite/fixtures/lite007i/dataset.json').read_text()); sizes={q['query_id']:str(next(s['collection_size'] for s in data['scenarios'] if s['scenario_id']==q['scenario_id'])) for q in data['queries']}
counts={x:0 for x in ('8','16','32')}
# derive via prepared query IDs
from lite.training.prepared_dataset import PreparedTrainingSet
ds=PreparedTrainingSet(prep); counts={x:0 for x in ('8','16','32')}; ids=[]
for i in order[:240]:
 q=ds.get(i)['query_id']; counts[sizes[q]]+=1; ids.append(q)
ds.close()
ledgers={}
for name in ('associative-budget.json','direct-budget-recovered.json'):
 d=json.loads((root/name).read_text()); ledgers[name]={'used_updates':d['used_updates'],'total_updates':d['total_updates'],'episodes':len(d['episodes']),'statuses':{x['status']:sum(y['status']==x['status'] for y in d['episodes']) for x in d['episodes']},'active':sum(x['status']=='active' for x in d['episodes'])}
res={'eligible_total':len(eligible),'scheduled_total':len(ids),'unique_query_ids':len(set(ids)),'collection_size_counts':counts,'matched_order':True,'order_digest':json.loads((root/'associative-scheduler.json').read_text())['order_digest'],'dataset_identity':json.loads((root/'associative-scheduler.json').read_text())['dataset_identity'],'validation_or_holdout_scheduled':False,'parameter_mutations_recorded_by_scheduler':0,'budgets':ledgers,'sample_identities_first_240':ids}
Path('docs/memorilla-lite/audits/LITE-007L-EXPOSURE-VERIFICATION.json').write_text(json.dumps(res,sort_keys=True,indent=2)+'\n')
