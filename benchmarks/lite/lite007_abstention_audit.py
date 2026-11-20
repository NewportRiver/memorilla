"""LITE-007 Stage A: audit abstention semantics without changing LITE-005 fixtures."""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from lite.encoder.onnx_encoder import MiniLMEncoder
from lite.encoder.manifest import MODEL_REPO, MODEL_REVISION, MODEL_SHA256
from lite.memory.provenance import SourceRecord, EncoderContract
from lite.memory.store import MemoryStore
from lite.retrieval import EvidenceCatalog, LexicalRetriever, SemanticRetriever, HybridRetriever, RetrievalController

FIX=ROOT/'benchmarks/lite/fixtures'

def contract():
    return EncoderContract(MODEL_REPO, MODEL_REVISION, MODEL_SHA256,
                           'MiniLMTokenizer-basic-v1', 'attention-mask-mean-l2-v1', 384)

def eligible(records, q):
    return [r for r in records if r.collection_id == q['scenario_id']
            and r.access_scope == q['access_scope']
            and (q['historical_cutoff'] is None or r.source_timestamp <= q['historical_cutoff'])]

def classify(q, records, out):
    req=set(q['required_evidence_ids'])
    elig=eligible(records,q)
    eligible_req=req & {r.evidence_id for r in elig}
    ids=[x['evidence_id'] for x in out.get('results',[])]
    policy_expected=(bool(req) and not eligible_req) or (not elig and not req)
    if not elig and not req:
        expected_kind='policy_ineligible_collection'
    elif policy_expected:
        expected_kind='policy_ineligible_required_evidence'
    elif q['expected_abstention'] and q['category']=='missing' and not req:
        expected_kind='insufficient_evidence'
    elif q['expected_abstention'] and not req:
        expected_kind='fixture_label_incorrect_answerable'
    elif q['expected_abstention']:
        expected_kind='fixture_label_or_logic_inconsistent'
    else:
        expected_kind='not_expected_abstention'
    unsupported_authorized=(expected_kind=='insufficient_evidence' and bool(ids)
                            and not req.intersection(ids))
    actual_abstain=not ids
    return {
        'query_id':q['query_id'], 'category':q['category'], 'query':q['query'],
        'fixture_expected_abstention':bool(q['expected_abstention']),
        'policy_expected_abstention':policy_expected,
        'expected_kind':expected_kind, 'required_evidence_ids':sorted(req),
        'eligible_ids':[r.evidence_id for r in elig], 'returned_ids':ids,
        'actual_abstention':actual_abstain, 'status':out.get('status'),
        'reason':out.get('reason'), 'fallback_used':bool(out.get('fallback_used',False)),
        'unsupported_authorized_return':unsupported_authorized,
        'omitted':out.get('omitted',[]),
    }

def run():
    raw=json.loads((FIX/'development_sources.json').read_text())['sources']
    queries=json.loads((FIX/'development_queries.json').read_text())['queries']
    records=[SourceRecord.from_content(x['evidence_id'],x['content'],provenance=x['provenance'],source_revision=x['source_revision'],access_scope=x['access_scope'],source_timestamp=x['source_timestamp'],collection_id=x['collection_id'],chunk_id=x['chunk_id'],chunk_start=x['chunk_start'],chunk_end=x['chunk_end']) for x in raw]
    cat=EvidenceCatalog(records); store=MemoryStore(':memory:',cat.resolve); c=contract(); enc=MiniLMEncoder(threads=2); cache={}
    def encode(text):
        if text not in cache: cache[text]=enc.encode(text)
        return cache[text]
    for r in records: store.ingest(r,c,encode,'float16')
    sem=SemanticRetriever(store,enc,c,dtype='float16')
    controllers={'lexical':RetrievalController(LexicalRetriever(cat),sem,HybridRetriever(LexicalRetriever(cat),sem),cat),
                 'semantic':RetrievalController(LexicalRetriever(cat),sem,HybridRetriever(LexicalRetriever(cat),sem),cat),
                 'hybrid':RetrievalController(LexicalRetriever(cat),sem,HybridRetriever(LexicalRetriever(cat),sem),cat)}
    rows={m:[] for m in controllers}
    for q in queries:
        for method, controller in controllers.items():
            out=controller.retrieve(q['query'],method=method,collection_id=q['scenario_id'],access_scope=q['access_scope'],as_of=q['historical_cutoff'],top_k=5)
            rows[method].append(classify(q,records,out))
    summary={}
    for method, items in rows.items():
        counts=Counter(x['expected_kind'] for x in items)
        policy=[x for x in items if x['policy_expected_abstention']]
        answerable=[x for x in items if x['expected_kind']=='fixture_label_incorrect_answerable']
        insufficient=[x for x in items if x['expected_kind']=='insufficient_evidence']
        summary[method]={
            'rows':len(items), 'expected_kind_counts':dict(sorted(counts.items())),
            'policy_abstentions_expected':len(policy),
            'policy_abstentions_correct':sum(x['actual_abstention'] for x in policy),
            'answerable_mislabeled_rows':len(answerable),
            'answerable_mislabeled_actual_abstentions':sum(x['actual_abstention'] for x in answerable),
            'insufficient_evidence_rows':len(insufficient),
            'unsupported_authorized_returns':sum(x['unsupported_authorized_return'] for x in insufficient),
            'fallback_events':sum(x['fallback_used'] for x in items),
            'resolved_but_irrelevant_not_counted_as_unsupported_before':sum(x['unsupported_authorized_return'] for x in insufficient),
        }
    report={'schema_version':1,'milestone':'LITE-007','stage':'A','original_fixture_unchanged':True,
            'development_query_count':len(queries),'development_source_count':len(records),
            'holdout_labels_accessed':False,'summary':summary,'rows':rows,
            'interpretation':{
                'policy_abstention':'No eligible required evidence because access scope or historical eligibility blocked it.',
                'insufficient_evidence':'The query asks for explicitly unrecorded evidence and has no required evidence IDs.',
                'fixture_label_incorrect_answerable':'The fixture marks abstention but the historical records contain answerable evidence.',
                'unsupported_authorized_return':'The resolver verified returned source identity/scope, but the result is not a required or supporting record for an insufficient-evidence query.',
                'unsupported_verified_claim':'Not measured by source resolution alone; factual support must be scored separately from resolver validity.'}}
    store.close(); cat.close(); enc.close(); return report

if __name__=='__main__': print(json.dumps(run(),sort_keys=True,indent=2))
