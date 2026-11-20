import json
from pathlib import Path

import numpy as np
import pytest

from lite.training.prepared_dataset import PreparedTrainingSet, prepare_training_artifact

DATA = Path(__file__).parents[2] / 'benchmarks/lite/fixtures/lite007'


def test_prepared_artifact_is_train_only_and_lazy(tmp_path):
    artifact = tmp_path / 'train.npz'
    meta = prepare_training_artifact(DATA, artifact)
    prepared = PreparedTrainingSet(artifact)
    assert meta['partition'] == 'train'
    assert meta['holdout_included'] is False
    assert len(prepared.eligible_indices()) == 715
    example = prepared.get(prepared.eligible_indices()[0])
    assert example['positive_mask'].any()
    assert example['valid_mask'].sum() > example['positive_mask'].sum()
    prepared.close()


def test_prepared_artifact_rejects_corruption_and_wrong_partition(tmp_path):
    artifact = tmp_path / 'train.npz'
    prepare_training_artifact(DATA, artifact)
    raw = artifact.read_bytes()
    artifact.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match='checksum'):
        PreparedTrainingSet(artifact)
    prepare_training_artifact(DATA, artifact)
    meta_path = artifact.with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    meta['partition'] = 'validation'
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='training partition'):
        PreparedTrainingSet(artifact)
