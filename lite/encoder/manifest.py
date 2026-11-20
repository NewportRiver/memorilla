"""Pinned model/runtime provenance for the single approved encoder artifact."""
from __future__ import annotations
from pathlib import Path
MODEL_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_REPO='sentence-transformers/all-MiniLM-L6-v2'
MODEL_LICENSE='Apache-2.0'
MODEL_ARTIFACT='onnx/model_qint8_arm64.onnx'
MODEL_SHA256='4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474'
MODEL_BYTES=23026053
OUTPUT_DIM=384
MAX_INPUT_TOKENS=128
CACHE_DIR=Path.home()/'.cache/memorilla-lite/minilm-1110a243'
def model_paths(cache_dir=None):
 p=Path(cache_dir) if cache_dir else CACHE_DIR
 return {'model':p/MODEL_ARTIFACT,'vocab':p/'vocab.txt','config':p/'config.json','sentence_config':p/'sentence_bert_config.json','tokenizer_config':p/'tokenizer_config.json','special_tokens':p/'special_tokens_map.json'}
