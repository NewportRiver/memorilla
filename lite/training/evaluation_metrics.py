from __future__ import annotations

from typing import Any


def ranking_metrics(rows, cutoffs=(1, 3, 5)) -> dict[str, Any]:
    """Compute denominator-explicit retrieval metrics from ranked IDs and required IDs.

    Each row may contain: ranked_ids, required_ids, answerable, unsupported_ids,
    temporal_error, policy_violation. Abstention rows have answerable=False.
    """
    rows = list(rows)
    answerable = [r for r in rows if r.get('answerable', bool(r.get('required_ids')))]
    abstention = [r for r in rows if not r.get('answerable', bool(r.get('required_ids')))]
    result = {'examples': len(rows), 'answerable_denominator': len(answerable), 'abstention_denominator': len(abstention)}
    for k in cutoffs:
        hit = []; recall = []; complete = []
        for r in answerable:
            required = set(r.get('required_ids', ())); retrieved = required & set(r.get('ranked_ids', ())[:k])
            hit.append(bool(retrieved)); recall.append(len(retrieved) / len(required) if required else 0.0); complete.append(bool(required) and retrieved == required)
        result[f'hit_rate_at_{k}'] = sum(hit) / len(hit) if hit else None
        result[f'recall_at_{k}'] = sum(recall) / len(recall) if recall else None
        result[f'complete_required_evidence_at_{k}'] = sum(complete) / len(complete) if complete else None
    rr = []
    coverage = []
    unsupported = 0
    false_abstention = 0
    for r in answerable:
        ranked = list(r.get('ranked_ids', ()))
        required = set(r.get('required_ids', ()))
        positions = [i + 1 for i, item in enumerate(ranked) if item in required]
        rr.append(1.0 / min(positions) if positions else 0.0)
        coverage.append(len(set(ranked) & required) / len(required) if required else 0.0)
        unsupported += sum(item in set(r.get('unsupported_ids', ())) for item in ranked)
    for r in abstention:
        if r.get('ranked_ids'):
            false_abstention += 1
    result['mrr'] = sum(rr) / len(rr) if rr else None
    result['multi_evidence_coverage'] = sum(coverage) / len(coverage) if coverage else None
    result['complete_required_evidence'] = sum(x == 1.0 for x in coverage) / len(coverage) if coverage else None
    result['false_abstention'] = false_abstention
    result['abstention_accuracy'] = (len(abstention) - false_abstention) / len(abstention) if abstention else None
    result['unsupported_evidence_retrieval'] = unsupported / len(answerable) if answerable else None
    result['temporal_error_count'] = sum(bool(r.get('temporal_error', False)) for r in rows)
    result['policy_violation_count'] = sum(bool(r.get('policy_violation', False)) for r in rows)
    return result
