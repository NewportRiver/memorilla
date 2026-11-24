"""Prepare the immutable LITE-008K synthetic development set and MiniLM features."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from lite.encoder.onnx_encoder import MiniLMEncoder

SEED=20261124
ROOT=Path.home()/'.cache/memorilla-lite/LITE-008K'
VALUES=['red','blue','green','yellow','purple','orange']
ENTITIES=['cobalt','linen','marble','cedar','orbit','quartz','velvet','harbor','meadow','signal','lantern','willow','granite','saffron','ripple','thistle','plume','canyon']
TEMPLATES=['What color belongs to entity {entity}?','Which color is assigned to entity {entity}?','Name the color recorded for entity {entity}.']

def sha(path):
 h=hashlib.sha256();
 with path.open('rb') as f:
  while chunk:=f.read(1<<20): h.update(chunk)
 return h.hexdigest()

def main():
 ROOT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(SEED)
 perm=rng.permutation(len(ENTITIES)); rows=[]
 for i,eidx in enumerate(perm):
  rows.append({'id':int(i),'entity':ENTITIES[int(eidx)],'value':VALUES[int(rng.integers(len(VALUES)))],'template':TEMPLATES[i%len(TEMPLATES)],'record_text':f'Memory record: entity {ENTITIES[int(eidx)]} has color {VALUES[int(rng.integers(len(VALUES)))]}.'})
 # Correct record text must use its assigned value; make it explicit and deterministic.
 for r in rows: r['record_text']=f'Memory record: entity {r["entity"]} has color {r["value"]}.'
 splits={'train':list(range(0,8)),'calibration':list(range(8,12)),'development':list(range(12,18))}
 entities={k:[rows[i]['entity'] for i in v] for k,v in splits.items()}
 overlap={f'{a}_{b}':sorted(set(entities[a])&set(entities[b])) for a in splits for b in splits if a<b}
 manifest={'schema':'LITE-008K-dataset-v1','seed':SEED,'values':VALUES,'templates':TEMPLATES,'rows':rows,'partitions':splits,'entity_overlap':overlap,'fact_overlap':{},'answer_in_query':False,'protected_final_holdout_accessed':False,'distractor_policy':'unrelated entity and conflicting-value records are generated in controls'}
 (ROOT/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 enc=MiniLMEncoder(threads=1); features=enc.encode_batch([r['record_text'] for r in rows]).astype(np.float32); identity=enc.identity(); enc.close()
 np.save(ROOT/'features.npy',features); meta={'schema':'LITE-008K-features-v1','shape':list(features.shape),'dtype':str(features.dtype),'encoder':identity,'sha256':sha(ROOT/'features.npy'),'source_manifest_sha256':sha(ROOT/'dataset-manifest.json')};(ROOT/'features.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(json.dumps({'manifest':str(ROOT/'dataset-manifest.json'),'features':str(ROOT/'features.npy'),'feature_sha256':meta['sha256'],'encoder':identity}))
if __name__=='__main__': main()
