"""LITE-008P Q4 transfer diagnostic using the existing qualified harness."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;CACHE=Path.home()/'.cache/memorilla-lite/LITE-008P';MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R4/runtime/SmolLM2-135M-Q4_K_M.gguf';HARNESS=Path.home()/'.cache/memorilla-lite/LITE-008G/embedding_harness';TOK=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
def one(name,r,slot,tok,a,b):
 ids=[tok.eos_token_id]+tok('Question: '+r['query']+' Answer:',add_special_tokens=True)['input_ids'];base=Path('/data/data/com.termux/files/home/.cache/memorilla-lite/LITE-008P')/name;np.asarray(ids,dtype='<i4').tofile(str(base)+'.ids');emb=np.zeros((len(ids),576),np.float32);emb[0]=slot;np.asarray(emb,dtype='<f4').tofile(str(base)+'.f32');np.asarray([a,b],dtype='<i4').tofile(str(base)+'.target');out=Path(str(base)+'.json');cmd=[str(HARNESS),'--model',str(MODEL),'--ids',str(base)+'.ids','--embd',str(base)+'.f32','--targets',str(base)+'.target','--out',str(out),'--mode','embd','--ctx','256','--steps','0','--threads','1'];st=time.monotonic();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);peak=0
 while p.poll() is None:
  try:peak=max(peak,int(Path(f'/proc/{p.pid}/status').read_text().split('VmRSS:')[1].split()[0])*1024)
  except:pass
  time.sleep(.01)
 stdout,stderr=p.communicate();d=json.loads(out.read_text()) if out.exists() else {'error':stderr[-500:]};d.update({'condition':name,'exit':p.returncode,'seconds':time.monotonic()-st,'artifact':str(MODEL),'sampled_peak_rss_bytes':peak,'stderr':stderr[-300:]});return d
def main():
 t=AutoTokenizer.from_pretrained(str(TOK),local_files_only=True,use_fast=True);man=json.loads((ROOT/'dataset-manifest.json').read_text());res=[]
 for cid,fn in [('case_08','fresh-oracle-08-09.json'),('case_09','fresh-oracle-08-09.json'),('case_10','fresh-oracle-10-11.json'),('case_11','fresh-oracle-10-11.json')]:
  r=next(x for x in man['rows'] if x['case_id']==cid);rr=json.loads((CACHE/fn).read_text())['results'];a=next(x for x in rr if x['case_id']==cid and x['target']=='A');b=next(x for x in rr if x['case_id']==cid and x['target']=='B');ia=t(' '+r['answer_a'],add_special_tokens=False)['input_ids'][0];ib=t(' '+r['answer_b'],add_special_tokens=False)['input_ids'][0];res += [one(cid+'-A',r,np.asarray(a['final_slots'],np.float32),t,ia,ib),one(cid+'-B',r,np.asarray(b['final_slots'],np.float32),t,ia,ib)]
 out={'schema':'LITE-008P-quantized-transfer-v1','status':'completed','artifact':str(MODEL),'results':res,'protected_final_holdout_accessed':False};(ROOT/'quantized-transfer.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'results':len(res),'exits':[x['exit'] for x in res]}))
if __name__=='__main__':main()
