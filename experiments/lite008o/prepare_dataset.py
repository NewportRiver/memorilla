"""Create the LITE-008O randomized counterfactual oracle manifest."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
VALUES=['amber','violet','copper','silver','cedar','coral','linen','marble']
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'dataset-manifest.json'
def main():
 rng=random.Random(20261124);entities=[f'quartz_{rng.randrange(10**10):010d}' for _ in range(16)];rng.shuffle(entities);rows=[]
 for i in range(16):
  a,b=VALUES[i%8],VALUES[(i+3)%8];
  if i%2: a,b=b,a
  order=['A','B'] if i%2==0 else ['B','A']
  e=entities[i];q=f'Which stored value belongs to record {e}?';other=f'quartz_{rng.randrange(10**10):010d}'
  rows.append({'case_id':f'case_{i:02d}','partition':'calibration' if i<8 else 'oracle','entity':e,'query':q,'answer_a':a,'answer_b':b,'candidate_order':order,'relevant_evidence':f'Record {e} stores value {a}.','counterfactual_evidence':f'Record {e} stores value {b}.','unrelated_evidence':f'Record {other} stores value {a}.','eligibility_rule':{'query_only_abs_margin_max':4.0,'relevant_correct_margin_min':1.0,'counterfactual_reversal_margin_max':-1.0,'unrelated_margin_min':-1.0}})
 payload={'schema':'LITE-008O-counterfactual-oracle-v1','seed':20261124,'values':VALUES,'rows':rows,'calibration_case_ids':[r['case_id'] for r in rows[:8]],'oracle_case_ids':[r['case_id'] for r in rows[8:]],'partitions_entity_disjoint':True,'primary_candidate_token_lengths':{v:1 for v in VALUES},'protected_final_holdout_accessed':False}
 payload['manifest_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();OUT.write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'path':str(OUT),'sha256':payload['manifest_sha256'],'rows':len(rows)}))
if __name__=='__main__':main()
