import json

import pytest

from lite.training.orchestration import EpisodeBudget


def test_episode_budget_persists_across_instances(tmp_path):
    path = tmp_path / 'budget.json'
    first = EpisodeBudget(path, 8, 'dataset-a', 'checkpoint-a')
    first.begin_episode('ep-1', 3)
    first.record_update('ep-1')
    first.record_update('ep-1')
    first.finish_episode('ep-1', 'stopped')
    second = EpisodeBudget(path, 8, 'dataset-a', 'checkpoint-a')
    assert second.state['used_updates'] == 2
    assert second.remaining_updates == 6
    with pytest.raises(ValueError):
        second.begin_episode('ep-2', 7)


def test_episode_budget_rejects_identity_mismatch(tmp_path):
    path = tmp_path / 'budget.json'
    EpisodeBudget(path, 8, 'dataset-a', 'checkpoint-a')
    with pytest.raises(ValueError):
        EpisodeBudget(path, 8, 'dataset-b', 'checkpoint-a')


def test_episode_budget_is_atomic_json(tmp_path):
    path = tmp_path / 'budget.json'
    budget = EpisodeBudget(path, 2, 'd', 'c')
    budget.begin_episode('ep', 1)
    budget.record_update('ep')
    payload = json.loads(path.read_text())
    assert payload['used_updates'] == 1
    assert not list(tmp_path.glob('*.tmp'))
