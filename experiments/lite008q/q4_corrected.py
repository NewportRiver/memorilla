"""Corrected independent Q4 candidate scoring for P optimized embeddings."""
import json,os,subprocess,time,hashlib
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;CACHE=Path.home()/'.cache/memorilla-lite/LITE-008P';OUT=Path('/data/data/com.termux/files/home/.cache/memorilla-lite/LITE-008Q');MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R4/runtime/SmolLM2-135M-Q4_K_M.gguf';HARNESS=Path.home()/'.cache/memorilla-lite/LITE-008G/embedding_harness';TOK=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
def run(name,r,slot,tok,target):
 OUT.mkdir(parents=True,exist_ok=True);base=OUT/name;ids=[tok.eos_token_id]+tok('Question: '+r['query']+' Answer:',add_special_tokens=True)['input_ids'];np.asarray(ids,dtype='<i4').tofile(str(base)+'.ids');emb=np.zeros((len(ids),576),np.float32);emb[0]=slot;emb.astype('<f4').tofile(str(base)+'.f32');np.asarray([target],dtype='<i4').tofile(str(base)+'.target');out=Path(str(base)+'.json');cmd=[str(HARNESS),'--model',str(MODEL),'--ids',str(base)+'.ids','--embd',str(base)+'.f32','--targets',str(base)+'.target','--out',str(out),'--mode','embd','--ctx','256','--steps','0','--threads','1'];p=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True);st=time.monotonic();rss=pss=0;avail=None
 while p.poll() is None:
  try:rss=max(rss,int(Path(f'/proc/{p.pid}/status').read_text().split('VmRSS:')[1].split()[0])*1024)
  except:pass
  try:pss=max(pss,int(Path(f'/proc/{p.pid}/smaps_rollup').read_text().split('Pss:')[1].split()[0])*1024)
  except:pass
  try:avail=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
  except:pass
  time.sleep(.01)
 p.wait();d=json.loads(out.read_text()) if out.exists() else {'error':'no output'};d.update({'name':name,'target_id':target,'exit':p.returncode,'seconds':time.monotonic()-st,'peak_rss_bytes':rss,'peak_pss_bytes':pss,'minimum_available_bytes':avail,'target_token_count':1});return d
def main():
 t=AutoTokenizer.from_pretrained(str(TOK),local_files_only=True,use_fast=True);man=json.loads((Path('/data/data/com.termux/files/home/memorilla-lite-workspace/experiments/lite008p/dataset-manifest.json')).read_text());res=[]
 for cid,fn in [('case_08','fresh-oracle-08-09.json'),('case_09','fresh-oracle-08-09.json'),('case_10','fresh-oracle-10-11.json'),('case_11','fresh-oracle-10-11.json')]:
  r=next(x for x in man['rows'] if x['case_id']==cid);rr=json.loads((CACHE/fn).read_text())['results'];
  for target_name,x in [('A',next(x for x in rr if x['case_id']==cid and x['target']=='A')),('B',next(x for x in rr if x['case_id']==cid and x['target']=='B'))]:
   a=t(' '+r['answer_a'],add_special_tokens=False)[0] if False else t(' '+r['answer_a'],add_special_tokens=False)['input_ids'][0];b=t(' '+r['answer_b'],add_special_tokens=False)['input_ids'][0];res.append({'case_id':cid,'direction':target_name,'fp32_margin':None,'a_id':a,'b_id':b,'a':run(f'{cid}-{target_name}-A',r,np.asarray(x['final_slots'],np.float32),t,a),'b':run(f'{cid}-{target_name}-B',r,np.asarray(x['final_slots'],np.float32),t,b)})
 for x in res:
  x['q4_logp_a']=x['a']['target_logprobs'][0];x['q4_logp_b']=x['b']['target_logprobs'][0];x['q4_margin']=x['q4_logp_a']-x['q4_logp_b'];x['q4_nll_a']=-x['q4_logp_a'];x['q4_nll_b']=-x['q4_logp_b']
 out={'schema':'LITE-008Q-corrected-q4-transfer-v1','status':'completed','scoring':'independent single-token runs from identical query/memory state; no sequential candidate comparison','artifact':str(MODEL),'tokenizer':str(TOK),'results':res,'artifact_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'protected_final_holdout_accessed':False};(ROOT/'q4-corrected-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'cases':len(res),'margins':[round(x['q4_margin'],3) for x in res]}))
if __name__=='__main__':main()
