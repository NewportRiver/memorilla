"""Bounded LITE-008G synthetic semantic-conditioning evaluation."""
from __future__ import annotations
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from gguf import GGUFReader,dequantize
from lite.encoder.onnx_encoder import MiniLMEncoder
from experiments.lite008f.projection import LinearEmbeddingBridge

SEED=20261124
ROOT=Path.home()/'.cache/memorilla-lite/LITE-008G'
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R4/runtime/SmolLM2-135M-Q4_K_M.gguf'
HARNESS=ROOT/'embedding_harness'
TOKDIR=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
VALUES=['red','blue','green','yellow','purple','orange','white','black']
TEMPLATES=['What is the code for {entity}? Answer:','Which code belongs to {entity}? Answer:']

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  while b:=f.read(1<<20): h.update(b)
 return h.hexdigest()

def run_monitored(cmd):
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 peak=0
 while p.poll() is None:
  try:
   s=Path(f'/proc/{p.pid}/status').read_text(); peak=max(peak,int(s.split('VmRSS:')[1].split()[0])*1024)
  except (FileNotFoundError,IndexError): pass
  time.sleep(.01)
 stdout,stderr=p.communicate()
 return p.returncode,stdout,stderr,peak

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(SEED)
 entities=[f'person {w} {i:02d}' for i,w in enumerate(['cobalt','linen','marble','cedar','orbit','quartz','velvet','harbor','meadow','signal','lantern','willow','granite','saffron','ripple','thistle','plume','canyon','silver','maple','comet','tundra','pepper','island','raven','copper','violet','forest','sunset','anchor','canvas','breeze'])]
 perm=rng.permutation(len(entities)); values=[VALUES[i%len(VALUES)] for i in perm]
 rows=[{'id':i,'entity':entities[i],'value':values[i],'template':TEMPLATES[i%2]} for i in range(len(entities))]
 splits={'train':list(range(16)),'calibration':list(range(16,24)),'development':list(range(24,32))}
 manifest={'schema':'LITE-008G-dataset-v1','seed':SEED,'value_set':VALUES,'templates':TEMPLATES,'rows':rows,'partitions':splits,'protected_final_holdout_accessed':False}
 (ROOT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 tok=AutoTokenizer.from_pretrained(str(TOKDIR),local_files_only=True,use_fast=True)
 rr=GGUFReader(str(MODEL)); tensor=next(t for t in rr.tensors if t.name=='token_embd.weight'); emb=dequantize(tensor.data,tensor.tensor_type).astype(np.float32)
 encoder=MiniLMEncoder(threads=1)
 texts=[f'Remember: {r["entity"]} has code {r["value"]}.' for r in rows]
 features=encoder.encode_batch(texts).astype(np.float32)
 np.save(ROOT/'features.npy',features)
 def answer_ids(v):
  ids=tok(' '+v,add_special_tokens=False)['input_ids']
  if len(ids)!=1: raise RuntimeError((v,ids))
  return ids
 targets=np.vstack([emb[answer_ids(r['value'])[0]] for r in rows]).astype(np.float32)
 base=LinearEmbeddingBridge(seed=SEED)
 hist=LinearEmbeddingBridge(seed=SEED); hist.train_sgd_legacy_unscaled(features[splits['train']],targets[splits['train']],300,.08)
 corr=LinearEmbeddingBridge(seed=SEED); corr.train_sgd(features[splits['train']],targets[splits['train']],300,.08*576/2)
 untrained=base
 constant=targets[splits['train']].mean(axis=0)
 def prompt_ids(r, evidence=None):
  text=(f'Memory: {r["entity"]} has code {evidence}. ' if evidence else '')+r['template'].format(entity=r['entity'])
  return tok(text,add_special_tokens=True)['input_ids']
 def write_case(name, r, latent=None, evidence=None):
  ids=prompt_ids(r,evidence); rows_emb=emb[np.asarray(ids,dtype=np.int64)].astype(np.float32)
  if latent is not None: ids=[ids[0]]+ids; rows_emb=np.vstack([latent,rows_emb])
  ip=ROOT/f'{name}.ids'; ep=ROOT/f'{name}.f32'; tp=ROOT/f'{name}.target'
  np.asarray(ids,dtype='<i4').tofile(ip); rows_emb.astype('<f4').tofile(ep); np.asarray(answer_ids(r['value']),dtype='<i4').tofile(tp)
  mode='embd' if latent is not None else 'token'
  out=ROOT/f'{name}.json'; cmd=[str(HARNESS),'--model',str(MODEL),'--ids',str(ip),'--embd',str(ep),'--targets',str(tp),'--out',str(out),'--mode',mode,'--ctx','256','--steps','0','--threads','1']
  t=time.monotonic(); rc,stdout,stderr,rss=run_monitored(cmd); d=json.loads(out.read_text()) if rc==0 else {'error':stderr[-500:]}
  d['score_peak_rss_bytes']=rss
  genout=ROOT/f'{name}.greedy.json'
  gcmd=[str(HARNESS),'--model',str(MODEL),'--ids',str(ip),'--embd',str(ep),'--out',str(genout),'--mode',mode,'--ctx','256','--steps','1','--threads','1']
  grc,gstdout,gstderr,grss=run_monitored(gcmd)
  if grc==0: d['greedy_generated']=json.loads(genout.read_text()).get('generated',[])
  else: d['greedy_error']=gstderr[-500:]
  d['generation_peak_rss_bytes']=grss
  d.update({'condition':name,'exit':rc,'duration_seconds':time.monotonic()-t,'prompt_token_count':len(ids),'command':' '.join(cmd),'generation_command':' '.join(gcmd)})
  return d
 results=[]; counters=[]
 # calibration capability prerequisite with fixed prompt format
 for i in splits['calibration']:
  r=rows[i]
  for cond,ev in [('query_only',None),('relevant_text',r['value']),('irrelevant_text',VALUES[(VALUES.index(r['value'])+1)%len(VALUES)]),('counterfactual_text',VALUES[(VALUES.index(r['value'])+2)%len(VALUES)])]:
   results.append(write_case(f'cal-{i}-{cond}',r,evidence=ev))
 # development controls, all one latent slot and same query boundary
 for i in splits['development']:
  r=rows[i]; x=features[i:i+1]
  fact=hist.forward(x)[0]; factc=corr.forward(x)[0]; un=untrained.forward(x)[0]
  other=hist.forward(features[[splits['development'][0] if i!=splits['development'][0] else splits['development'][1]]])[0]
  cf_text=VALUES[(VALUES.index(r['value'])+1)%len(VALUES)]
  conds={'query_only':None,'zero':np.zeros(576,np.float32),'random':np.random.default_rng(SEED+i).normal(0,.02,576).astype(np.float32),'untrained':un,'historical':fact,'corrected':factc,'mismatched':other,'counterfactual_projected':hist.forward(encoder.encode(f'Remember: {r["entity"]} has code {cf_text}.').reshape(1,-1))[0],'constant_mean':constant,'oracle':emb[answer_ids(r['value'])[0]]}
  for c,z in conds.items(): results.append(write_case(f'dev-{i}-{c}',r,latent=z))
  # counterfactual direction: same question, projected fact A versus B
  for val,label in [(r['value'],'A'),(cf_text,'B')]:
   z=hist.forward(encoder.encode(f'Remember: {r["entity"]} has code {val}.').reshape(1,-1))[0]
   d=write_case(f'cf-{i}-{label}',r,latent=z); d['supplied_value']=val; counters.append(d)
 (ROOT/'semantic-raw.json').write_text(json.dumps(results,indent=2)+'\n')
 (ROOT/'counterfactual-raw.json').write_text(json.dumps(counters,indent=2)+'\n')
 print(json.dumps({'dataset':str(ROOT/'dataset-manifest.json'),'results':len(results),'counterfactual_pairs':len(counters)//2,'features_sha256':sha(ROOT/'features.npy'),'model_sha256':sha(MODEL)}))
if __name__=='__main__': main()
