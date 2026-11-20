"""Best-effort process/system telemetry; unavailable values remain None."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import os,resource,time
from .preflight import available_ram_bytes,swap_bytes,storage_free_bytes

def rss_bytes(pid=None):
    pid=os.getpid() if pid is None else pid
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'): return int(line.split()[1])*1024
    except (OSError,ValueError): pass
    return None

def thread_count(pid=None):
    pid=os.getpid() if pid is None else pid
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('Threads:'): return int(line.split()[1])
    except (OSError,ValueError): pass
    return None

def cpu_load_1m():
    try: return float(Path('/proc/loadavg').read_text().split()[0])
    except (OSError,PermissionError,ValueError): return None

@dataclass
class Snapshot:
    label:str
    elapsed_ms:float
    rss_bytes:int|None
    peak_rss_bytes:int|None
    available_ram_bytes:int|None
    swap_total_bytes:int|None
    swap_free_bytes:int|None
    storage_free_bytes:int|None
    threads:int|None
    cpu_load_1m:float|None
    def as_dict(self): return asdict(self)

def snapshot(label, started_ns, storage_path='.'):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; keep the platform assumption explicit.
    peak=int(usage.ru_maxrss*1024)
    st,sf=swap_bytes()
    try: disk=storage_free_bytes(storage_path)
    except OSError: disk=None
    return Snapshot(label,(time.monotonic_ns()-started_ns)/1e6,rss_bytes(),peak,available_ram_bytes(),st,sf,disk,thread_count(),cpu_load_1m())
