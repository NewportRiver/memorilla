"""Create fresh LITE-008P calibration and oracle entities."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent;ROOT.mkdir(parents=True,exist_ok=True);rng=random.Random(20261125)
values=['amber','violet','copper','silver','cedar','coral','linen','marble'];rows=[];used=set()
for i in range(16):
 entity=f'feldspar_{rng.randrange(10**9,10**10)}'
 while entity in used: entity=f'feldspar_{rng.randrange(10**9,10**10)}'
 used.add(entity);a,b=rng.sample(values,2);order=['A','B'] if i%2==0 else ['B','A']; rows.append({'case_id':f'case_{i:02d}','partition':'calibration' if i<8 else 'oracle','entity':entity,'query':f'Which stored value belongs to record {entity}?','answer_a':a,'answer_b':b,'candidate_order':order,'relevant_evidence':f'Record {entity} stores value {a}.','counterfactual_evidence':f'Record {entity} stores value {b}.','unrelated_evidence':f'Record feldspar_{rng.randrange(10**9,10**10)} stores value {a}.'})
manifest={'schema':'LITE-008P-fresh-counterfactual-v1','seed':20261125,'values':values,'rows':rows,'calibration_case_ids':[r['case_id'] for r in rows[:8]],'oracle_case_ids':[r['case_id'] for r in rows[8:]],'entity_disjoint':True,'primary_token_lengths':{v:1 for v in values},'protected_final_holdout_accessed':False}
(ROOT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'rows':len(rows),'sha256':hashlib.sha256((ROOT/'dataset-manifest.json').read_bytes()).hexdigest()}))
if __name__=='__main__':pass
