"""LITE-008H bounded capability and low-dimensional SPSA diagnostic."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from gguf import GGUFReader,dequantize
from experiments.lite008h.monitor import run_monitored

SEED=20261124
ROOT=Path.home()/'.cache/memorilla-lite/LITE-008H'; ROOT.mkdir(parents=True,exist_ok=True)
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R4/runtime/SmolLM2-135M-Q4_K_M.gguf'
HARNESS=Path.home()/'.cache/memorilla-lite/LITE-008G/embedding_harness'
TOKDIR=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
VALUES=['red','blue','green','yellow']

def main():
 rng=np.random.default_rng(SEED)
 entities=[f'entity {w} {i:02d}' for i,w in enumerate(['cobalt','linen','marble','cedar','orbit','quartz','velvet','harbor','meadow','signal','lantern','willow','granite','saffron','ripple','thistle','plume','canyon','silver','maple','comet','tundra','pepper','island'])]
 values=[VALUES[i%4] for i in rng.permutation(len(entities))]
 rows=[{'id':i,'entity':entities[i],'value':values[i]} for i in range(len(entities))]
 splits={'train':list(range(8)),'calibration':list(range(8,16)),'development':list(range(16,24))}
 manifest={'schema':'LITE-008H-dataset-v1','seed':SEED,'values':VALUES,'rows':rows,'partitions':splits,'protected_final_holdout_accessed':False,'notes':'Values intentionally shared; entity/fact partitions disjoint.'}
 (ROOT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 tok=AutoTokenizer.from_pretrained(str(TOKDIR),local_files_only=True,use_fast=True)
 reader=GGUFReader(str(MODEL)); tensor=next(t for t in reader.tensors if t.name=='token_embd.weight'); matrix=dequantize(tensor.data,tensor.tensor_type).astype(np.float32)
 def tids(text): return tok(text,add_special_tokens=True)['input_ids']
 def answer_ids(value): return tok(' '+value,add_special_tokens=False)['input_ids']
 assert all(len(answer_ids(v))==1 for v in VALUES)
 calls=[]
 def invoke(name,prompt,z=None,target=None,generations=0):
  ids=tids(prompt); emb=matrix[np.asarray(ids,dtype=np.int64)].astype(np.float32); mode='token'
  if z is not None: ids=[ids[0]]+ids;emb=np.vstack([z,emb]);mode='embd'
  stem=ROOT/name; np.asarray(ids,dtype='<i4').tofile(stem.with_suffix('.ids'));emb.astype('<f4').tofile(stem.with_suffix('.f32'))
  if target is not None: np.asarray(target,dtype='<i4').tofile(stem.with_suffix('.target'))
  out=stem.with_suffix('.json');cmd=[str(HARNESS),'--model',str(MODEL),'--ids',str(stem.with_suffix('.ids')),'--embd',str(stem.with_suffix('.f32')),'--out',str(out),'--mode',mode,'--ctx','256','--steps',str(generations),'--threads','1']
  if target is not None:cmd += ['--targets',str(stem.with_suffix('.target')),'--steps','0']
  tele=run_monitored(cmd,20.0);d=json.loads(out.read_text()) if tele['exit_status']==0 and out.exists() else {'error':'native failure'};d.update({'name':name,'prompt':prompt,'telemetry':tele,'command':' '.join(cmd)});calls.append(d);return d
 def score(name,prompt,z,value): return -invoke(name,prompt,z,answer_ids(value))['mean_nll']
 def prompt(r,evidence=None):return (f'Memory: {r["entity"]} has code {evidence}. ' if evidence else '')+f'What is the code for {r["entity"]}? Answer:'
 capability=[]
 for i in splits['calibration']:
  r=rows[i];alt=VALUES[(VALUES.index(r['value'])+1)%4]
  for label,ev in [('query',None),('relevant',r['value']),('irrelevant',alt),('counterfactual',alt)]:
   p=prompt(r,ev);correct=score(f'cap-{i}-{label}-correct',p,None,r['value']);wrong=score(f'cap-{i}-{label}-wrong',p,None,alt);g=invoke(f'cap-{i}-{label}-greedy',p,None,None,1);capability.append({'id':i,'condition':label,'correct_logprob':correct,'alternate_logprob':wrong,'margin':correct-wrong,'greedy':g.get('generated',[]),'telemetry':g['telemetry']})
 capout={'schema':'LITE-008H-capability-v1','rows':capability,'candidate_ranking_accuracy':{c:float(np.mean([x['margin']>0 for x in capability if x['condition']==c])) for c in ['query','relevant','irrelevant','counterfactual']},'raw_calls':len(calls)}
 (ROOT/'capability-results.json').write_text(json.dumps(capout,indent=2)+'\n')
 # Oracle/SPSA diagnostic only after calibration; one example, 16 coefficients.
 dim=16; basis=rng.normal(size=(dim,576)).astype(np.float32);basis/=np.linalg.norm(basis,axis=1,keepdims=True)
 train=rows[splits['train'][0]]; alt=VALUES[(VALUES.index(train['value'])+1)%4];q=prompt(train); eps=.05;lr=.4;theta=np.zeros(dim,np.float32)
 def zof(t):return (t@basis).astype(np.float32)
 def objective(t,tag):return score(tag+'-c',q,zof(t),train['value'])-score(tag+'-a',q,zof(t),alt)
 trace=[]
 for it in range(8):
  delta=rng.choice([-1.,1.],size=dim).astype(np.float32);plus=objective(theta+eps*delta,f'spsa-{it}-p');minus=objective(theta-eps*delta,f'spsa-{it}-m');g=(plus-minus)/(2*eps)*delta;theta=(theta+lr*g).astype(np.float32);trace.append({'iteration':it,'plus':plus,'minus':minus,'theta_norm':float(np.linalg.norm(theta))})
 opt=zof(theta);
 # Budget-matched random search over eight candidate vectors using the same objective.
 random_scores=[]
 for j in range(8): random_scores.append(objective(rng.normal(size=dim).astype(np.float32),f'random-{j}'))
 oracle={'schema':'LITE-008H-oracle-v1','train_entity':train,'alternate_value':alt,'parameterization':'z = theta @ fixed_basis','coefficients':dim,'basis_shape':[dim,576],'trace':trace,'initial_margin':objective(np.zeros(dim,np.float32),'oracle-initial'),'optimized_margin':objective(theta,'oracle-final'),'random_search_margins':random_scores,'decoder_evaluations':len(calls)}
 (ROOT/'oracle-results.json').write_text(json.dumps(oracle,indent=2)+'\n')
 print(json.dumps({'calls':len(calls),'capability':capout['candidate_ranking_accuracy'],'oracle':oracle}))
if __name__=='__main__':main()
