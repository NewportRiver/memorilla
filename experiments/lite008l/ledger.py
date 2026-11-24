"""Durable append-only phase ledger for LITE-008L experiments."""
from __future__ import annotations
import json, os, time, uuid
from pathlib import Path

ROOT=Path.home()/'.cache/memorilla-lite/LITE-008L'
LEDGER=ROOT/'experiment-ledger.jsonl'

def record(phase: str, status: str, **fields):
    ROOT.mkdir(parents=True,exist_ok=True)
    event={'event_id':uuid.uuid4().hex,'timestamp_unix':time.time(),'experiment':'LITE-008L', 'phase':phase,'status':status,**fields}
    with LEDGER.open('a',encoding='utf-8') as f:
        f.write(json.dumps(event,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    return event

def phase_start(phase: str, **fields): return record(phase,'started',**fields)
def phase_end(phase: str, **fields): return record(phase,'completed',**fields)
def phase_abort(phase: str, **fields): return record(phase,'aborted',**fields)
