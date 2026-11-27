import json,hashlib,re,random
from pathlib import Path
from transformers import AutoTokenizer
from experiments.lite008p.p_experiment import MODEL
ROOT=Path(__file__).resolve().parents[1]; AUD=ROOT/'docs/memorilla-lite/audits'; OUT=AUD/'LITE-008X-DATASET-MANIFEST.json'; prior=json.loads((AUD/'LITE-008W-R2-DATASET-MANIFEST.json').read_text())
def norm(s): return re.sub(r'[^a-z0-9]+','',s.lower())
def prior_words():
 s=' '.join(json.dumps(prior).split()); return {norm(x) for x in re.findall(r'[A-Za-z]+',s)}
def main():
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True); old=prior_words()
 candidates=['cobalt','saffron','granite','walrus','bronze','tundra','otter','crimson','silk','panda','copper','violet','quartzite','badger','indigo','basalt','falcon','ambergris','teal','ivory','cedarwood','topaz','harbor','meadow','lantern','pebble','willow','salmon','ochre','marten','plum','coral','bison','linenette','sable','nickel','bronzeleaf','moss','violetblue','canyon','delta','island','orchid','falconry','tulip','mustard','onyx','ruby','platinum','citrine','spruce','maple','raven','marigold','pearl','scarlet','azure','geode','beacon','cactus','dolphin','elm','granule','pumice','rhodium']
 vals=[]; rejected=[]
 for v in candidates:
  n=norm(v); ids=tok(' '+v,add_special_tokens=False)['input_ids']; morph=any(n.startswith(x) or x.startswith(n) for x in old if len(x)>3)
  if n in old or morph or len(ids)!=1: rejected.append({'value':v,'normalized':n,'token_ids':ids,'reason':'prior/morphological collision' if morph or n in old else 'not_single_token'}); continue
  vals.append(v)
 if len(vals)<24: raise RuntimeError(f'only {len(vals)} isolated values: {vals}')
 # deterministic allocations; values are absent from all W manifest text and single-token under qualified tokenizer
 groups={'B':vals[0:16],'C':vals[16:32]}
 familiar=['linen','amber','marble','cedar']; rows=[]; rng=random.Random(8008)
 def add(group,i,a,b,ea,eb,entities=None,entity_name=None,query_override=None):
  cid=f'x{group.lower()}_{i:02d}'; entity_name=entity_name or f'xentity_{group.lower()}_{i:02d}'; q=query_override or f'What value does record {entity_name} store?'
  row={'id':cid,'group':group,'partition':'lite008x_evaluation','entity':entity_name,'query':q,'answer_a':a,'answer_b':b,'evidence_a':ea,'evidence_b':eb,'unrelated_evidence':f'Record unrelated_{i} stores value {b}.','entities':entities or [f'xentity_{group.lower()}_{i:02d}'],'candidate_token_ids':{'a':tok(' '+a,add_special_tokens=False)['input_ids'],'b':tok(' '+b,add_special_tokens=False)['input_ids']},'sequence_token_lengths':{'query':len(tok(q,add_special_tokens=True)['input_ids']),'evidence_a':len(tok(q+' '+ea,add_special_tokens=True)['input_ids']),'evidence_b':len(tok(q+' '+eb,add_special_tokens=True)['input_ids'])}}
  if max(row['sequence_token_lengths'].values())>32: raise RuntimeError(row)
  rows.append(row)
 for i in range(8):
  a,b=familiar[(2*i)%4],familiar[(2*i+1)%4]; ent=f'xentity_a_{i:02d}'; add('A',i,a,b,f'Record {ent} stores value {a}.',f'Record {ent} stores value {b}.')
 for i in range(8):
  a,b=groups['B'][2*i],groups['B'][2*i+1]; ent=f'xentity_b_{i:02d}'; add('B',i,a,b,f'Record {ent} stores value {a}.',f'Record {ent} stores value {b}.')
 templates=[('The archive entry {e} is assigned the value {v}.','The archive entry {e} is assigned the value {v}.='),('For identifier {e}, the recorded attribute is {v}.','For identifier {e}, the recorded attribute is {v}.'),('A ledger labels {e} with {v}.','A ledger labels {e} with {v}.')]
 for i in range(8):
  a,b=groups['C'][2*i],groups['C'][2*i+1]; ent=f'xentity_c_{i:02d}'; t=templates[i%len(templates)][0]; add('C',i,a,b,t.format(e=ent,v=a),t.format(e=ent,v=b))
 for i in range(8):
  a,b=groups['B'][2*i],groups['B'][2*i+1]; qa=f'd{i:02d}a'; qb=f'd{i:02d}b'; ea=f'{qa}={a}; {qb}={b}.'; eb=f'{qa}={b}; {qb}={a}.'; add('D',i,a,b,ea,eb,[qa,qb],qa,f'Value {qa}?')
 for r in rows:
  r['content_hash']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 out={'schema':'LITE-008X-dataset-v1','status':'FROZEN','seed':8008,'manifest_source':'new deterministic content; no protected holdout','groups':{g:{'pairs':8,'values':(familiar if g=='A' else groups.get(g,groups['B']))} for g in 'ABCD'},'vocabulary_isolation':{'familiar_values':familiar,'unfamiliar_values':groups,'prior_normalized_words_count':len(old),'rejected_candidates':rejected,'single_token_required':True},'rows':rows,'candidate_generation_decisions':{'candidate_pool_size':len(candidates),'accepted_values':vals,'rejected':rejected},'optimizer_updates':0,'manifest_sha256_pending':True}; OUT.write_text(json.dumps(out,indent=2)+'\n'); out['manifest_sha256']=hashlib.sha256(OUT.read_bytes()).hexdigest(); OUT.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'rows':len(rows),'accepted_values':len(vals),'sha256':out['manifest_sha256']}))
if __name__=='__main__':main()
