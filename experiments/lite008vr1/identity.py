"""Fail-closed identity bindings for historical calibration diagnostics."""
import hashlib
import json
from pathlib import Path

FIELDS = ('manifest_sha256', 'partition', 'entity', 'evidence', 'query',
          'candidate_answers', 'candidate_token_ids', 'direction', 'model',
          'generation', 'vector_sha256', 'case_id', 'purpose')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def seal(identity):
    if set(identity) != set(FIELDS):
        raise ValueError('FAIL_CLOSED: incomplete or unexpected identity fields')
    if identity['partition'] != 'calibration' or identity['purpose'] != 'historical_reconstruction_diagnostic':
        raise ValueError('FAIL_CLOSED: partition/purpose')
    if identity['direction'] not in ('A', 'B'):
        raise ValueError('FAIL_CLOSED: direction')
    return {'identity': identity, 'identity_sha256': digest(identity)}


def validate(expected, supplied, vector):
    for record in (expected, supplied):
        identity = record['identity']
        seal(identity)
        if record['identity_sha256'] != digest(identity):
            raise ValueError('FAIL_CLOSED: identity checksum')
    if expected != supplied:
        raise ValueError('FAIL_CLOSED: originating identity mismatch')
    if len(vector) != 576 or digest(vector) != expected['identity']['vector_sha256']:
        raise ValueError('FAIL_CLOSED: vector hash/dimensions')
    return True
