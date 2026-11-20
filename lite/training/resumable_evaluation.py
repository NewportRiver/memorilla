from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ResumableEvaluation:
    def __init__(self, root, evaluation_id, model_identity, dataset_identity, partition, query_ids, batch_size=32):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'state.json'; self.results_path = self.root / 'results.jsonl'
        self.identity = {'evaluation_id': evaluation_id, 'model_identity': model_identity, 'dataset_identity': dataset_identity, 'partition': partition, 'query_ids': list(query_ids), 'batch_size': batch_size}
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get('identity') != self.identity: raise ValueError('evaluation identity mismatch')
            self.state = state
        else:
            self.state = {'identity': self.identity, 'status': 'pending', 'next_index': 0, 'completed_batches': [], 'result_ids': []}
            self._save()

    def _save(self):
        tmp = self.state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(self.state, sort_keys=True, indent=2) + '\n'); os.replace(tmp, self.state_path)

    def _existing_result_ids(self):
        if not self.results_path.exists(): return set()
        return {json.loads(line)['result_id'] for line in self.results_path.read_text().splitlines() if line.strip()}

    def run(self, evaluate_batch, fail_after_batches=None):
        if self.state['status'] == 'complete': return self.aggregate()
        self.state['status'] = 'running'; self._save(); batch_number = len(self.state['completed_batches'])
        existing = self._existing_result_ids()
        while self.state['next_index'] < len(self.identity['query_ids']):
            start = self.state['next_index']; ids = self.identity['query_ids'][start:start + self.identity['batch_size']]
            rows = evaluate_batch(ids)
            if len(rows) != len(ids): raise ValueError('batch result count mismatch')
            serialized = []
            for query_id, row in zip(ids, rows):
                result_id = _digest({'evaluation_id': self.identity['evaluation_id'], 'model_identity': self.identity['model_identity'], 'query_id': query_id})
                if result_id in existing: raise ValueError('duplicate evaluation result')
                serialized.append({'result_id': result_id, 'query_id': query_id, 'value': row})
            with self.results_path.open('a', encoding='utf-8') as handle:
                for row in serialized: handle.write(json.dumps(row, sort_keys=True) + '\n')
                handle.flush(); os.fsync(handle.fileno())
            existing.update(x['result_id'] for x in serialized); self.state['result_ids'] = sorted(existing); self.state['next_index'] = start + len(ids); self.state['completed_batches'].append({'start': start, 'count': len(ids), 'digest': _digest(serialized)}); self._save(); batch_number += 1
            if fail_after_batches is not None and batch_number >= fail_after_batches: raise RuntimeError('evaluation fault injection after atomic batch')
        self.state['status'] = 'complete'; self._save(); return self.aggregate()

    def aggregate(self):
        if self.state['status'] != 'complete': raise RuntimeError('partial evaluation cannot be aggregated')
        rows = [json.loads(line) for line in self.results_path.read_text().splitlines() if line.strip()]
        if len(rows) != len(self.identity['query_ids']) or len({x['result_id'] for x in rows}) != len(rows): raise ValueError('incomplete or duplicate evaluation results')
        return {'evaluation_id': self.identity['evaluation_id'], 'model_identity': self.identity['model_identity'], 'dataset_identity': self.identity['dataset_identity'], 'partition': self.identity['partition'], 'denominator': len(rows), 'results': rows}
