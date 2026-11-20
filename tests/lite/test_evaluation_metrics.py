from lite.training.evaluation_metrics import ranking_metrics


def test_ranking_metrics_reports_denominators_and_abstention():
    result = ranking_metrics([
        {'ranked_ids': ['a', 'x'], 'required_ids': ['a'], 'answerable': True, 'unsupported_ids': ['x']},
        {'ranked_ids': [], 'required_ids': [], 'answerable': False},
        {'ranked_ids': ['z'], 'required_ids': [], 'answerable': False},
    ])
    assert result['examples'] == 3
    assert result['answerable_denominator'] == 1
    assert result['abstention_denominator'] == 2
    assert result['recall_at_1'] == 1.0
    assert result['multi_evidence_coverage'] == 1.0
    assert result['unsupported_evidence_retrieval'] == 1.0
    assert result['false_abstention'] == 1
    assert result['abstention_accuracy'] == 0.5
