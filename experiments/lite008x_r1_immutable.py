"""Experimental frozen projection evaluator; no fitting or decoder loading.

JSON test/reference adapter only. Historical PT artifacts require separately
qualified decoding and complete provenance; do not convert/reconstruct them here.
Callers must provide independently trusted expected hashes and complete model
file inventories (weights, configuration, tokenizer and runtime settings).
"""
import hashlib
import json
import math
import struct
from pathlib import Path


class IntegrityError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def verify_bundle(spec):
    """Exact inventory prevents omitted weight shards and tokenizer sidecars."""
    if set(spec) != {'root', 'files', 'settings'} or not spec['files'] or not spec['settings']:
        raise IntegrityError('incomplete model identity')
    root = Path(spec['root']).resolve()
    actual = {}
    for p in root.rglob('*'):
        if p.is_symlink():
            raise IntegrityError('symlink in model identity')
        if p.is_file():
            actual[p.relative_to(root).as_posix()] = file_hash(p)
    if actual != spec['files']:
        raise IntegrityError('model inventory/content mismatch')
    return digest(canonical({'files': actual, 'settings': spec['settings']}))


def f32(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise IntegrityError('nonfinite/non-numeric tensor')
    try:
        result = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as exc:
        raise IntegrityError('float32 overflow') from exc
    if not math.isfinite(result):
        raise IntegrityError('float32 overflow')
    return result


def vector(values, size):
    if not isinstance(values, list) or len(values) != size:
        raise IntegrityError('tensor shape mismatch')
    return [f32(v) for v in values]


REQUIRED = {'manifest_sha256', 'partition', 'entity', 'query', 'evidence',
            'candidate_answers', 'candidate_token_ids', 'factual_direction',
            'encoder_identity', 'decoder_identity', 'generation_config'}


def evaluate(path, expected_sha256, shape, encoder, decoder, identity, feature,
             expected_input_sha256, *, _after_load=None):
    """One slot, tiny reference matvec; returns hashes only after postchecks.

    This is not a production BLAS implementation or a latent acceptance test.
    A trusted caller supplies shape and expected bytes rather than learning them
    from the artifact under evaluation. No solver, training, or callback scorer.
    """
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise IntegrityError('expected artifact SHA required')
    if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape) or shape[0]*shape[1] > 1048576:
        raise IntegrityError('invalid/bounded shape')
    if Path(path).stat().st_size > 67108864:
        raise IntegrityError('artifact exceeds reference size cap')
    raw = Path(path).read_bytes()
    if digest(raw) != expected_sha256:
        raise IntegrityError('artifact SHA mismatch')
    enc, dec = verify_bundle(encoder), verify_bundle(decoder)
    if not REQUIRED <= identity.keys() or any(identity[k] is None or identity[k] == '' for k in REQUIRED):
        raise IntegrityError('incomplete content identity')
    if identity['encoder_identity'] != enc or identity['decoder_identity'] != dec:
        raise IntegrityError('input/model identity mismatch')
    if identity['factual_direction'] not in ('A', 'B') or len(identity['candidate_answers']) != 2 or len(identity['candidate_token_ids']) != 2:
        raise IntegrityError('invalid candidate/direction identity')
    bound_input = digest(canonical({'identity': identity, 'feature': feature}))
    if bound_input != expected_input_sha256:
        raise IntegrityError('input hash mismatch')
    obj = json.loads(raw)
    if obj.get('dtype') != 'float32' or obj.get('shape') != list(shape):
        raise IntegrityError('artifact dtype/shape mismatch')
    if obj.get('encoder_identity') != enc or obj.get('decoder_identity') != dec:
        raise IntegrityError('artifact/model identity mismatch')
    if len(obj['W']) != shape[0]:
        raise IntegrityError('matrix shape mismatch')
    w = [vector(row, shape[1]) for row in obj['W']]
    eos = vector(obj['eos'], shape[0]); x = vector(feature, shape[1])
    z = [f32(e + sum(a*b for a, b in zip(row, x))) for e, row in zip(eos, w)]
    if _after_load is not None:  # Fault injection hook, not a scientific scorer.
        _after_load()
    if file_hash(path) != expected_sha256 or verify_bundle(encoder) != enc or verify_bundle(decoder) != dec:
        raise IntegrityError('post-evaluation content changed')
    return {'status': 'REFERENCE_MATVEC_ONLY', 'input_sha256': bound_input,
            'projection_sha256': expected_sha256, 'output_sha256': digest(canonical(z)),
            'output': z, 'encoder_identity': enc, 'decoder_identity': dec,
            'optimizer_updates': 0, 'solver_operations': 0, 'decoder_forwards': 0}
