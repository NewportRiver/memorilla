import argparse,json,hashlib
from pathlib import Path
from experiments.lite008w.work_units import run_direction
from lite.encoder.onnx_encoder import MiniLMEncoder
ROOT=Path(__file__).resolve().parents[2]; AUD=ROOT/'docs/memorilla-lite/audits'; OUT=Path.home()/'.cache/memorilla-lite/LITE-008W-R3/teachers'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--target',choices=['A','B'],required=True);a=ap.parse_args(); d=json.loads((AUD/'LITE-008W-R2-DATASET-MANIFEST.json').read_text()); r=next(x for x in d['rows'] if x['id']==a.id); r=dict(r);r['target']=a.target;r['evidence']=r['evidence_a' if a.target=='A' else 'evidence_b']; manifest_sha=hashlib.sha256((AUD/'LITE-008W-R2-DATASET-MANIFEST.json').read_bytes()).hexdigest(); enc=MiniLMEncoder(threads=1); eid=enc.identity();enc.close(); print(json.dumps(run_direction(r,a.target,manifest_sha,eid,OUT)))
if __name__=='__main__':main()
