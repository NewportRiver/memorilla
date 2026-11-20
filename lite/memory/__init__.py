"""Disposable, evidence-linked persistent embedding bank."""
from .provenance import SourceRecord,EncoderContract
from .quantization import encode_vector,decode_vector
from .store import MemoryStore
