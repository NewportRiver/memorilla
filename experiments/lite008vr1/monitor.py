"""Sample-enforced zero-update provenance preflight (not decoder execution)."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'docs/memorilla-lite/audits/LITE-008V-R1-RESOURCE-TELEMETRY.json'


def run():
    record = {'phase': 'provenance_recovery', 'optimizer_updates': 0, 'solver_operations': 0,
              'decoder_forwards': 0, 'rss_ceiling_bytes': 268435456,
              'reserve_bytes': 536870912, 'timeout_seconds': 300, 'samples': [],
              'enforcement': '50ms sampled child RSS/reserve; not kernel hard RSS enforcement',
              'status': 'preflight'}
    REPORT.write_text(json.dumps(record, indent=2)+'\n')
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    python = Path.home()/'.cache/memorilla-lite/LITE-008D-R4/venv/bin/python'
    start = time.monotonic()
    child = subprocess.Popen([str(python), '-m', 'experiments.lite008vr1.recover'], cwd=ROOT, env=env)
    violation = None
    while child.poll() is None:
        available = int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
        try:
            status = Path(f'/proc/{child.pid}/status').read_text()
            rss = int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1])*1024
        except (FileNotFoundError, StopIteration):
            rss = 0
        elapsed = time.monotonic()-start
        record['samples'].append({'seconds': elapsed, 'rss_bytes': rss, 'available_bytes': available})
        if rss > record['rss_ceiling_bytes'] or available < record['reserve_bytes'] or elapsed > 300:
            violation = 'rss_reserve_or_timeout'
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
            break
        time.sleep(.05)
    record.update(exit_code=child.wait(), violation=violation, wall_seconds=time.monotonic()-start,
                  status='completed' if not violation and child.returncode == 0 else 'blocked')
    record['peak_sampled_rss_bytes'] = max(x['rss_bytes'] for x in record['samples'])
    record['minimum_sampled_available_bytes'] = min(x['available_bytes'] for x in record['samples'])
    REPORT.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k != 'samples'}))


if __name__ == '__main__':
    run()
