from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path


def rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def available_ram_bytes():
    try:
        with open('/proc/meminfo', encoding='utf-8') as handle:
            values = {line.split(':', 1)[0]: int(line.split()[1]) * 1024 for line in handle if ':' in line}
        return values.get('MemAvailable', 0)
    except OSError:
        return 0


def phase_marker(path, phase, event='end', started_ns=None):
    row = {
        'phase': phase,
        'event': event,
        'timestamp_ns': time.time_ns(),
        'rss_bytes': rss_bytes(),
        'available_ram_bytes': available_ram_bytes(),
        'elapsed_ns': time.perf_counter_ns() - started_ns if started_ns else None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return row