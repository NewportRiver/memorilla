import json,hashlib,statistics
from pathlib import Path
import torch
repo=Path(__file__).resolve().parents[2];cache=Path.home()/'.cache/memorilla-lite/LITE-008P';u=cache/'lite008u-teachers.json';d=json.loads(u.read_text());man=json.loads((repo/'experiments/lite008q/dataset-manifest.json').read_text());teachers=[]
for r in d['results']:
 v=torch.tensor(r['final_slots'][0],dtype=torch.float32); b=json.dumps(v.tolist(),separators=(',',':')).encode();row=next(x for x in man['rows'] if x['case_id']==r['case_id']);teachers.append({'case_id':r['case_id'],'target':r['target'],'source':str(u),'embedding_sha256':hashlib.sha256(b).hexdigest(),'shape':list(v.shape),'dtype':'float32','norm':float(v.norm()),'final_margin':float(r['trajectory'][-1]['margin']),'updates':r['updates'],'lr':r['lr'],'initialization':'EOS input embedding','decoder':'SmolLM2-135M LITE-008D-R3','encoder':'MiniLM LITE-003','candidate_answer':row['answer_a' if r['target']=='A' else 'answer_b'],'embedding':v.tolist()})
(repo/'experiments/lite008u/teacher-dataset.json').write_text(json.dumps({'schema':'LITE-008U-teacher-dataset-v1','status':'qualified','source':str(u),'teachers':teachers},indent=2)+'\n')
X=torch.stack([torch.tensor(t['embedding']) for t in teachers]);_,S,_=torch.linalg.svd(X-X.mean(0),full_matrices=False);by={}
for t,v in zip(teachers,X):by.setdefault(t['case_id'],{})[t['target']]=v
pairs=[float((z['A']-z['B']).norm()) for z in by.values()];same=[];diff=[]
for i in range(len(X)):
 for j in range(i+1,len(X)):
  c=float(torch.nn.functional.cosine_similarity(X[i:i+1],X[j:j+1]).item());(same if teachers[i]['target']==teachers[j]['target'] else diff).append(c)
energy=(S*S);cum=torch.cumsum(energy,0)/energy.sum();rank=int((cum<.99).sum().item()+1)
print(json.dumps({'pair_distances':pairs,'same_answer_cosine_mean':statistics.mean(same),'different_answer_cosine_mean':statistics.mean(diff),'singular_values':S.tolist(),'effective_rank_99':rank,'norm_range':[min(t['norm'] for t in teachers),max(t['norm'] for t in teachers)]}))