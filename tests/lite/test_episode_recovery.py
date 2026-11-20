import json
import multiprocessing as mp

import pytest

from lite.training.orchestration import EpisodeBudget


def _claim(path):
    budget = EpisodeBudget(path, 1, 'd', 'c')
    try:
        budget.begin_episode('worker', 1)
        return True
    except ValueError:
        return False


def test_concurrent_budget_claims_are_serialized(tmp_path):
    path = str(tmp_path / 'budget.json')
    EpisodeBudget(path, 1, 'd', 'c')
    with mp.Pool(2) as pool:
        results = pool.map(_claim, [path, path])
    assert sum(results) == 1
    assert json.loads(open(path).read())['episodes'][0]['requested_updates'] == 1


def test_budget_counts_interrupted_update_conservatively(tmp_path):
    budget = EpisodeBudget(tmp_path / 'budget.json', 2, 'd', 'c')
    budget.begin_episode('ep', 2)
    budget.record_update('ep')
    budget.finish_episode('ep', 'interrupted')
    resumed = EpisodeBudget(tmp_path / 'budget.json', 2, 'd', 'c')
    assert resumed.state['used_updates'] == 1
    with pytest.raises(ValueError):
        resumed.begin_episode('next', 2)
