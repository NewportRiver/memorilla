"""Read-only resource preflight for bounded Android experiments."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, shutil, time

MIB=1024*1024
@dataclass(frozen=True)
class ResourceLimits:
    min_available_ram_bytes:int=1024*MIB
    abort_available_ram_bytes:int=768*MIB
    target_rss_bytes:int=128*MIB
    max_rss_bytes:int=256*MIB
    max_storage_bytes:int=50*MIB
    max_threads:int=2
    max_concurrent_processes:int=1
    timeout_seconds:float=30.0

def meminfo() -> dict[str,int]:
    out={}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            k, v = line.split(':',1); parts=v.strip().split(); out[k]=int(parts[0])*(1024 if len(parts)>1 and parts[1]=='kB' else 1)
    except (OSError,ValueError):
        return {}
    return out

def available_ram_bytes() -> int|None:
    m=meminfo(); return m.get('MemAvailable')

def swap_bytes() -> tuple[int|None,int|None]:
    m=meminfo(); return m.get('SwapTotal'),m.get('SwapFree')

def storage_free_bytes(path: str|os.PathLike[str]) -> int:
    return shutil.disk_usage(path).free

def preflight(path='.', limits:ResourceLimits|None=None, requested_storage_bytes=0) -> dict:
    limits=limits or ResourceLimits()
    ram=available_ram_bytes(); free=storage_free_bytes(path)
    reasons=[]
    if ram is None: reasons.append('MemAvailable unavailable')
    elif ram < limits.min_available_ram_bytes: reasons.append('available RAM below minimum')
    if free < requested_storage_bytes: reasons.append('requested storage exceeds free storage')
    if requested_storage_bytes > limits.max_storage_bytes: reasons.append('requested storage exceeds configured maximum')
    return {'ok':not reasons,'reasons':reasons,'available_ram_bytes':ram,'storage_free_bytes':free,'limits':limits.__dict__.copy(),'timestamp_ns':time.time_ns()}

def assert_preflight(**kwargs):
    result=preflight(**kwargs)
    if not result['ok']: raise RuntimeError('; '.join(result['reasons']))
    return result
