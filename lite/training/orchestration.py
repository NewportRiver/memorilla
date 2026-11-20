from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class EpisodeBudget:
    """Atomic, persistent update budget shared across fresh training subprocesses."""

    def __init__(self, path, total_updates, dataset_identity, checkpoint_identity):
        self.path = Path(path)
        self.total_updates = int(total_updates)
        self.dataset_identity = str(dataset_identity)
        self.checkpoint_identity = str(checkpoint_identity)
        existed = self.path.exists()
        self.state = self._load_or_initialize()
        if not existed:
            self._commit()

    def _load_or_initialize(self):
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state['total_updates'] != self.total_updates or state['dataset_identity'] != self.dataset_identity or state['checkpoint_identity'] != self.checkpoint_identity:
                raise ValueError('episode budget identity mismatch')
            return state
        return {'schema_version': 1, 'total_updates': self.total_updates, 'used_updates': 0,
                'dataset_identity': self.dataset_identity, 'checkpoint_identity': self.checkpoint_identity,
                'episodes': []}

    def _commit(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=self.path.name + '.', dir=str(self.path.parent))
        os.close(fd)
        try:
            Path(temp).write_text(json.dumps(self.state, sort_keys=True, indent=2) + '\n')
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def begin_episode(self, episode_id, requested_updates):
        requested_updates = int(requested_updates)
        if requested_updates <= 0 or self.state['used_updates'] + requested_updates > self.total_updates:
            raise ValueError('episode exceeds persistent authorization budget')
        row = {'episode_id': str(episode_id), 'requested_updates': requested_updates, 'completed_updates': 0, 'status': 'active'}
        self.state['episodes'].append(row)
        self._commit()
        return row.copy()

    def record_update(self, episode_id):
        rows = [x for x in self.state['episodes'] if x['episode_id'] == str(episode_id) and x['status'] == 'active']
        if len(rows) != 1:
            raise ValueError('unknown or inactive episode')
        row = rows[0]
        if row['completed_updates'] >= row['requested_updates']:
            raise ValueError('episode update budget exhausted')
        row['completed_updates'] += 1
        self.state['used_updates'] += 1
        self._commit()

    def finish_episode(self, episode_id, status='completed'):
        rows = [x for x in self.state['episodes'] if x['episode_id'] == str(episode_id) and x['status'] == 'active']
        if len(rows) != 1:
            raise ValueError('unknown or inactive episode')
        rows[0]['status'] = str(status)
        self._commit()

    @property
    def remaining_updates(self):
        return self.total_updates - self.state['used_updates']
