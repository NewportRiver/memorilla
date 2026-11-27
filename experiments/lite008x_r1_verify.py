"""Bounded stdlib supervisor for X-R1's small engineering regressions only."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/memorilla-lite/audits/LITE-008X-R1-VERIFICATION.json'
PY = Path.home()/'.cache/memorilla-lite/LITE-008D-R4/venv/bin/python'


def available():
    return int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024


def main():
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='-1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
    command = [str(PY), '-m', 'pytest', '-q', '--confcutdir=tests/lite', 'tests/lite/test_lite008x_r1_immutable.py', 'tests/lite/test_lite008x.py', 'tests/lite/test_lite008w_identity.py', 'tests/lite/test_lite008w_r4_controls.py']
    record = {'command': command, 'kind': 'engineering fixtures; no scientific model execution', 'sampling_interval_seconds': .01, 'hard_timeout_seconds':300, 'scientific_seconds':0, 'optimizer_updates':0, 'solver_operations':0, 'decoder_forwards':0}
    proc = None; start = time.monotonic(); peak = 0; minimum = available(); reason = None
    try:
        if minimum < 536870912 or not PY.is_file():
            raise RuntimeError('preflight memory/interpreter blocked')
        with tempfile_log().open('w+') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                minimum = min(minimum, available())
                try:
                    status = Path(f'/proc/{proc.pid}/status').read_text()
                    if next(x for x in status.splitlines() if x.startswith('State:')).split()[1] == 'Z':
                        proc.wait(timeout=1)
                        break
                    rss = int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1])*1024
                    threads = int(next(x for x in status.splitlines() if x.startswith('Threads:')).split()[1])
                    children = []
                    # Android omits task/<pid>/children: inspect same-UID PPids.
                    for entry in Path('/proc').iterdir():
                        if not entry.name.isdigit(): continue
                        try:
                            if entry.stat().st_uid != os.getuid(): continue
                            child_status = (entry/'status').read_text()
                            parent = int(next(x for x in child_status.splitlines() if x.startswith('PPid:')).split()[1])
                            if parent == proc.pid: children.append(entry.name)
                        except FileNotFoundError:
                            continue  # raced with unrelated process exit
                except (FileNotFoundError, StopIteration):
                    if proc.poll() is not None: break
                    raise RuntimeError('resource telemetry unavailable')
                peak = max(peak, rss)
                if peak > 1073741824 or minimum < 536870912 or threads > 1 or children or time.monotonic()-start > 300:
                    raise RuntimeError('resource/thread/process/time guard')
                time.sleep(.01)
            proc.wait(timeout=1); log.seek(0); record['output'] = log.read(); record['returncode'] = proc.returncode
    except Exception as exc:
        reason = repr(exc)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
        record.update(status='PASS' if reason is None and record.get('returncode')==0 else 'BLOCKED_OR_FAILED', blocker=reason, peak_sampled_rss_bytes=peak, minimum_sampled_available_bytes=minimum, wall_seconds=time.monotonic()-start, child_reaped=proc is None or proc.poll() is not None, monitoring='10ms sampled; transient peaks between samples not certified')
        OUT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    return 0 if record['status']=='PASS' else 1


def tempfile_log():
    # Dedicated new addendum log; never overwrites original X artifacts.
    return ROOT/'docs/memorilla-lite/audits/LITE-008X-R1-TEST-OUTPUT.txt'


if __name__ == '__main__': raise SystemExit(main())
