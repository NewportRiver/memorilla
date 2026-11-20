from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .numpy_telemetry import available_ram_bytes, phase_marker, rss_bytes


def _checksum(state):
    import numpy as np
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(state[name]).tobytes())
    return digest.hexdigest()


def run_pilot(root, output_dir, max_updates=8, learning_rate=0.001, max_records=32, max_seconds=30.0,
              min_ram=1280 * 1024**2, abort_ram=1024 * 1024**2, max_rss=256 * 1024**2, authorized=True):
    if not authorized:
        raise PermissionError('NumPy training pilot requires explicit authorization')
    if int(os.environ.get('OMP_NUM_THREADS', '1')) != 1:
        raise RuntimeError('OMP_NUM_THREADS must be 1')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    telemetry_path = output / 'phases.jsonl'
    started = time.perf_counter_ns()
    phase_marker(telemetry_path, 'process_start', 'start', started)
    phase_marker(telemetry_path, 'numpy_import', 'start', started)
    import numpy as np
    phase_marker(telemetry_path, 'numpy_import', 'end', started)
    if available_ram_bytes() and available_ram_bytes() < min_ram:
        return {'status': 'blocked', 'reason': 'minimum_ram', 'updates': 0}
    phase_marker(telemetry_path, 'process_start', 'end', started)
    phase_marker(telemetry_path, 'dataset_loading', 'start', started)
    from lite.neural.config import NeuralConfig
    from lite.neural.numpy_model import NumpyAssociativeMemory
    from lite.neural.serialization import save_model
    from .dataset import load_development
    from .numpy_gradients import clip_gradients, scoring_head_gradients, trainable_names
    from .numpy_optimizer import sgd_step, snapshot_frozen, verify_frozen
    examples, data = load_development(Path(root), split='train')
    usable = [x for x in examples if x.positive_mask.any() and x.valid_mask.sum() > x.positive_mask.sum()]
    phase_marker(telemetry_path, 'dataset_loading', 'end', started)
    phase_marker(telemetry_path, 'model_initialization', 'start', started)
    model = NumpyAssociativeMemory(NeuralConfig(seed=20261120))
    initial = {k: v.copy() for k, v in model.state.items()}
    frozen = snapshot_frozen(model.state, trainable_names())
    phase_marker(telemetry_path, 'model_initialization', 'end', started)
    diagnostic = usable[0]
    _, diagnostic_latent = model.forward(diagnostic.query_embedding, diagnostic.memory_embeddings,
                                          diagnostic.valid_mask, return_latent=True)
    diagnostic_initial_loss = float(scoring_head_gradients(
        diagnostic_latent, diagnostic.memory_embeddings, model.state,
        diagnostic.positive_mask, diagnostic.valid_mask)[0])
    rows = []
    stop_reason = 'max_updates'
    for update in range(max_updates):
        if time.perf_counter_ns() - started > max_seconds * 1e9:
            stop_reason = 'time_limit'; break
        ram = available_ram_bytes()
        if ram and ram < abort_ram:
            stop_reason = 'abort_ram'; break
        if rss_bytes() > max_rss:
            stop_reason = 'rss_limit'; break
        example = usable[update % len(usable)]
        if len(example.evidence_ids) > max_records:
            example_memory = example.memory_embeddings[:max_records]
            example_ids = example.evidence_ids[:max_records]
            positive = np.asarray([x in set(example_ids) for x in example.evidence_ids[:max_records]], bool)
            valid = example.valid_mask[:max_records]
        else:
            example_memory, positive, valid = example.memory_embeddings, example.positive_mask, example.valid_mask
        phase_marker(telemetry_path, 'forward_and_gradient', 'start', started)
        _, latent = model.forward(example.query_embedding, example_memory, valid, return_latent=True)
        loss, _, grads = scoring_head_gradients(latent, example_memory, model.state, positive, valid)
        phase_marker(telemetry_path, 'forward_and_gradient', 'end', started)
        clipped, norm, scale = clip_gradients(grads, 1.0)
        phase_marker(telemetry_path, 'optimizer_execution', 'start', started)
        sgd_step(model.state, clipped, learning_rate)
        phase_marker(telemetry_path, 'optimizer_execution', 'end', started)
        rows.append({'update': update + 1, 'loss': float(loss), 'gradient_norm': norm, 'clip_scale': scale,
                     'rss_bytes': rss_bytes(), 'available_ram_bytes': available_ram_bytes(),
                     'elapsed_seconds': (time.perf_counter_ns() - started) / 1e9})
        if not verify_frozen(model.state, frozen):
            stop_reason = 'frozen_parameter_violation'; break
    phase_marker(telemetry_path, 'checkpoint_serialization', 'start', started)
    checkpoint = None
    if rows:
        checkpoint = save_model(model, output / 'lite007-numpy-score-head',
                                {'repo': 'all-MiniLM-L6-v2', 'revision': 'pinned', 'sha256': '4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474',
                                 'dimension': 384, 'contract': 'attention-mask-mean-l2-v1'}, 'numpy_scoring_head_sgd')
    phase_marker(telemetry_path, 'checkpoint_serialization', 'end', started)
    _, diagnostic_latent = model.forward(diagnostic.query_embedding, diagnostic.memory_embeddings,
                                          diagnostic.valid_mask, return_latent=True)
    diagnostic_final_loss = float(scoring_head_gradients(
        diagnostic_latent, diagnostic.memory_embeddings, model.state,
        diagnostic.positive_mask, diagnostic.valid_mask)[0])
    result = {'status': 'passed_pilot' if rows else 'blocked', 'stop_reason': stop_reason, 'updates': len(rows),
              'initial_loss': diagnostic_initial_loss, 'final_loss': diagnostic_final_loss,
              'diagnostic_query_id': diagnostic.query_id,
              'initial_weights_checksum': _checksum(initial), 'final_weights_checksum': _checksum(model.state),
              'trainable_parameters': list(trainable_names()), 'frozen_parameters_verified': verify_frozen(model.state, frozen),
              'telemetry': rows, 'checkpoint': checkpoint, 'peak_rss_bytes': rss_bytes(),
              'duration_seconds': (time.perf_counter_ns() - started) / 1e9, 'dataset_version': data.get('dataset_version')}
    (output / 'run.json.tmp').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    os.replace(output / 'run.json.tmp', output / 'run.json')
    return result