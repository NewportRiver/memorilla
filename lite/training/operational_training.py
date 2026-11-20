from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.neural.serialization import save_model, load_model
from .numpy_gradients import associative_gradients, clip_gradients
from .numpy_optimizer import sgd_step, snapshot_frozen, verify_frozen
from .orchestration import EpisodeBudget
from .prepared_dataset import PreparedTrainingSet
from .sample_scheduler import PersistentSampleScheduler
from .numpy_telemetry import available_ram_bytes, rss_bytes


def _state_checksum(state):
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode()); h.update(np.ascontiguousarray(state[name]).tobytes())
    return h.hexdigest()


def run_authorized_episode(prepared_artifact, output_dir, budget_path, checkpoint_identity,
                           requested_updates=8, learning_rate=0.001, max_records=32,
                           subset=None, authorized=False, authorization_total_updates=None,
                           min_ram=1280 * 1024**2, abort_ram=1024 * 1024**2,
                           max_rss=256 * 1024**2, max_seconds=30.0,
                           initial_checkpoint=None, scheduler_path=None, scheduler_seed=0, schedule_order=None):
    if not authorized:
        raise PermissionError('explicit episode authorization required')
    if str(checkpoint_identity).startswith(('LITE-007L', 'LITE-008')) and scheduler_path is None:
        raise ValueError('persistent scheduler is mandatory for corrected comparative experiments')
    if requested_updates > 8:
        raise ValueError('bounded episode maximum is eight updates')
    if os.environ.get('OMP_NUM_THREADS', '1') != '1':
        raise RuntimeError('OMP_NUM_THREADS must be 1')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); rss_start = rss_bytes()
    telemetry = output / 'phases.jsonl'
    def phase(name, status):
        with telemetry.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'phase': name, 'status': status, 'elapsed_seconds': time.perf_counter() - started, 'rss_bytes': rss_bytes(), 'available_ram_bytes': available_ram_bytes()}) + '\n')
            handle.flush(); os.fsync(handle.fileno())
    phase('process_start', 'start')
    ram = available_ram_bytes()
    if ram and ram < min_ram:
        phase('process_start', 'blocked_minimum_ram')
        raise RuntimeError('minimum RAM guard blocked episode')
    phase('dataset_loading', 'start'); prepared = PreparedTrainingSet(prepared_artifact); phase('dataset_loading', 'end')
    dataset_identity = prepared.metadata['content_sha256']
    budget = EpisodeBudget(budget_path, int(authorization_total_updates or requested_updates), dataset_identity, checkpoint_identity)
    episode_id = output.name
    budget.begin_episode(episode_id, requested_updates)
    phase('model_initialization', 'start')
    model = load_model(initial_checkpoint)[0] if initial_checkpoint is not None else NumpyAssociativeMemory(NeuralConfig(seed=20261120))
    phase('model_initialization', 'end')
    if subset is None:
        subset = tuple(name for name in model.state if name not in {'out_w', 'out_b'})
    frozen = snapshot_frozen(model.state, subset)
    usable = prepared.eligible_indices()
    scheduler = PersistentSampleScheduler(scheduler_path, dataset_identity, usable, scheduler_seed, schedule_order) if scheduler_path is not None else None
    episode_indices = scheduler.reserve(requested_updates, episode_id) if scheduler is not None else [usable[position % len(usable)] for position in range(requested_updates)]
    rows = []
    try:
        for position in range(requested_updates):
            if time.perf_counter() - started > max_seconds:
                raise TimeoutError('episode wall-time guard exceeded')
            ram = available_ram_bytes()
            if ram and ram < abort_ram:
                raise MemoryError('episode abort-RAM guard exceeded')
            if rss_bytes() > max_rss:
                raise MemoryError('episode RSS guard exceeded')
            phase('update', 'start')
            ex = prepared.get(episode_indices[position])
            mem = ex['memory_embeddings'][:max_records]; pos = ex['positive_mask'][:max_records]; valid = ex['valid_mask'][:max_records]
            phase('forward_backward', 'start'); loss, _, grads, cache = associative_gradients(model, ex['query_embedding'], mem, pos, valid); phase('forward_backward', 'end')
            selected = {name: grads[name] for name in subset if name in grads}
            clipped, norm, scale = clip_gradients(selected, 1.0)
            budget.record_update(episode_id)
            phase('optimizer', 'start'); sgd_step(model.state, clipped, learning_rate); phase('optimizer', 'end')
            if not verify_frozen(model.state, frozen):
                raise RuntimeError('frozen parameter violation')
            phase('checkpoint_serialization', 'start'); meta = save_model(model, output / 'checkpoint', {'dataset_identity': dataset_identity, 'checkpoint_identity': checkpoint_identity}, 'numpy_authorized_subset_sgd'); phase('checkpoint_serialization', 'end')
            rows.append({'position': position, 'query_id': ex['query_id'], 'loss': float(loss), 'gradient_norm': norm, 'clip_scale': scale, 'checkpoint_checksum': _state_checksum(model.state)})
            phase('update', 'end')
    except BaseException:
        budget.finish_episode(episode_id, 'interrupted')
        prepared.close()
        raise
    budget.finish_episode(episode_id, 'completed')
    prepared.close()
    result = {'status': 'passed_pilot', 'updates': len(rows), 'subset': list(subset), 'rows': rows, 'checkpoint': str(output / 'checkpoint'), 'final_checksum': _state_checksum(model.state), 'dataset_identity': dataset_identity, 'checkpoint_identity': checkpoint_identity, 'telemetry': {'absolute_peak_rss_bytes': rss_bytes(), 'rss_delta_bytes': max(0, rss_bytes() - rss_start), 'available_ram_bytes': available_ram_bytes(), 'wall_seconds': time.perf_counter() - started, 'numerical_threads': int(os.environ.get('OMP_NUM_THREADS', '1')), 'computational_subprocesses': 1, 'resource_guard_status': 'passed'}}
    (output / 'run.json.tmp').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    os.replace(output / 'run.json.tmp', output / 'run.json')
    return result
