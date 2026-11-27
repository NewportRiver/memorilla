"""Read-only bounded metadata audit; no tensor/model deserialization or solvers."""
import collections
import json
from pathlib import Path
from experiments.lite008x_r1_immutable import file_hash

ROOT=Path(__file__).resolve().parents[1]
AUD=ROOT/'docs/memorilla-lite/audits'
CACHE=Path.home()/'.cache/memorilla-lite'


def main():
    manifest=AUD/'LITE-008X-DATASET-MANIFEST.json'
    d=json.loads(manifest.read_text()); rows=d['rows']
    feature_path=CACHE/'LITE-008W-R2/features.json'
    fd=json.loads(feature_path.read_text())
    teachers=[]
    for path in sorted((CACHE/'LITE-008W-R3/teachers').glob('*.json')):
        meta=json.loads(path.read_text()); payload=meta['payload']
        matches=[x for x in fd['features'] if (x['id'],x['target'])==(payload['case_id'],payload['target'])]
        fields=['manifest_sha256','partition','entity','query','evidence','encoder_identity']
        teachers.append({'metadata':str(path),'metadata_sha256':file_hash(path),'artifact_sha256':file_hash(meta['artifact']), 'expected_sha256':meta['artifact_sha256'], 'artifact_matches':file_hash(meta['artifact'])==meta['artifact_sha256'], 'key':[payload['case_id'],payload['target']], 'candidate_feature_count':len(matches), 'feature_metadata_mismatches':[k for k in fields if len(matches)!=1 or matches[0][k]!=payload[k]], 'feature_manifest':matches[0]['manifest_sha256'] if len(matches)==1 else None,'teacher_manifest':payload['manifest_sha256'], 'decoder_has_weight_hash':any('weight' in k for k in payload['decoder_identity'])})
    projection=CACHE/'LITE-008W-R3/projection_selected.pt'
    expected=json.loads((AUD/'LITE-008W-R3-ARTIFACT-INTEGRITY.json').read_text())['projection_artifact_sha256']
    before=file_hash(projection)
    raw=[]
    for path in sorted((CACHE/'LITE-008X').glob('*.json')):
        obj=json.loads(path.read_text()); raw.append({'path':str(path),'sha256':file_hash(path),'rows':len(obj.get('rows',[])),'group':obj.get('group'),'keys':sorted(obj)})
    textual=json.loads((AUD/'LITE-008X-TEXTUAL-COMPETENCE-RESULTS.json').read_text())
    p=384; h=576; rank=8; n=1000
    estimates={'feature_dimension':p,'decoder_dimension_from_local_config':h,'dense_parameters':p*h,'dense_fp32_bytes':4*p*h,'S_float64_bytes':8*p*p,'T_float64_bytes':8*p*h,'S_T_W_bytes':8*p*p+8*p*h+4*p*h,'rank8_parameters':rank*(p+h),'rank8_fp32_bytes':4*rank*(p+h),'retrieval_1000_fp32_feature_bytes':4*n*p,'structured_example_d768_dense_fp32_bytes':4*768*h,'structured_example_d768_S_T_W_bytes':8*768*768+8*768*h+4*768*h,'teacher_1000_slot_fp32_bytes':4*n*h,'decoder_embedding_bf16_bytes':2*49152*h,'decoder_embedding_fp32_bytes':4*49152*h,'production_ceiling_bytes':268435456}
    originals={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(AUD.glob('LITE-008X-*')) if not p.name.startswith('LITE-008X-R1-')}
    out={'schema':'LITE-008X-R1-immutable-audit-v1','scope':'new engineering addendum; original R1 directive unavailable','baseline':'f2c30c5926aa2c355acb8af5b5e0eeecec4a9846','status':'PARTIAL_IDENTITY_BLOCKED','projection':{'path':str(projection),'expected_sha256':expected,'pre_sha256':before,'post_sha256':file_hash(projection),'matches_expected':before==expected,'tensor_deserialized':False,'numerical_evaluation':'BLOCKED: incomplete decoder content identity and cross-manifest feature binding; no PT-to-JSON conversion'},'features':{'path':str(feature_path),'sha256':file_hash(feature_path),'count':len(fd['features']),'exact_R3_training_feature_recovery':'NOT_CERTIFIED'},'teachers':teachers,'retained_raw_latent':raw,'raw_textual':{'summary_path':str(AUD/'LITE-008X-TEXTUAL-COMPETENCE-RESULTS.json'),'has_rows':'rows' in textual,'raw_source':textual.get('raw_source'),'cache_inventory':[p.name for p in sorted((CACHE/'LITE-008X').iterdir())],'status':'NOT_RECOVERED_IN_SCOPED_LOCATIONS'},'manifest':{'file_sha256':file_hash(manifest),'embedded_sha256':d.get('manifest_sha256'),'embedded_matches_final_file':d.get('manifest_sha256')==file_hash(manifest),'unrelated_answer_b_count':sum(r['unrelated_evidence'].endswith('value '+r['answer_b']+'.') for r in rows),'rows':len(rows),'A_pair_counts':dict(collections.Counter(r['answer_a']+' / '+r['answer_b'] for r in rows if r['group']=='A')),'D_queried_entities':[r['entity'] for r in rows if r['group']=='D']},'reference_evaluator':{'module':'experiments/lite008x_r1_immutable.py','adapter':'JSON float32 reference matvec only; not bitwise Torch FP32 accumulation','real_frozen_artifact_executed':False,'complete_model_inventory_required':True,'input_projection_output_hashes':True,'pre_post_hash_checks':True,'tests_report':'LITE-008X-R1-VERIFICATION.json'},'estimates':estimates,'original_X_report_hashes':originals,'counters':{'optimizer_updates':0,'teacher_generation':0,'projection_refits':0,'solver_operations':0,'decoder_forwards':0,'scientific_subprocesses':0,'scientific_seconds':0},'blocked_phases':['matched BF16/FP32 calibration','exact training feature SVD/subspace','real frozen projection forward','new benchmark tokenization and freezing','relational both-query execution','prospective latent acceptance (not authorized)']}
    (AUD/'LITE-008X-R1-IMMUTABLE-EVALUATOR-AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'teachers':len(teachers),'feature_manifest_mismatches':sum('manifest_sha256' in t['feature_metadata_mismatches'] for t in teachers),'projection_matches':before==expected,'estimates':estimates,'manifest':out['manifest'],'raw_latent':raw},indent=2))


if __name__=='__main__': main()
