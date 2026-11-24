"""Bounded child-process monitoring for LITE-008H/J."""
from __future__ import annotations
import os, signal, subprocess, time
from pathlib import Path


def _meminfo_bytes(key: str) -> int | None:
    try:
        text = Path('/proc/meminfo').read_text()
        line = next(line for line in text.splitlines() if line.startswith(key + ':'))
        return int(line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError):
        return None


def run_monitored(command: list[str], timeout_seconds: float, rss_ceiling_bytes: int | None = None,
                  available_reserve_bytes: int | None = None) -> dict:
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    peak_rss = peak_pss = 0
    min_available = None
    max_swap_used = 0
    timed_out = False
    resource_violation = None
    while True:
        rc = child.poll()
        status = Path(f'/proc/{child.pid}/status')
        if status.exists():
            try:
                text = status.read_text()
                peak_rss = max(peak_rss, int(text.split('VmRSS:')[1].split()[0]) * 1024)
            except (OSError, IndexError, ValueError):
                pass
        rollup = Path(f'/proc/{child.pid}/smaps_rollup')
        if rollup.exists():
            try:
                text = rollup.read_text()
                peak_pss = max(peak_pss, int(text.split('Pss:')[1].split()[0]) * 1024)
            except (OSError, IndexError, ValueError):
                pass
        available = _meminfo_bytes('MemAvailable')
        swap_total = _meminfo_bytes('SwapTotal') or 0
        swap_free = _meminfo_bytes('SwapFree') or 0
        if available is not None:
            min_available = available if min_available is None else min(min_available, available)
        max_swap_used = max(max_swap_used, swap_total - swap_free)
        if rss_ceiling_bytes is not None and peak_rss > rss_ceiling_bytes:
            resource_violation = 'rss_ceiling'
        if available_reserve_bytes is not None and available is not None and available < available_reserve_bytes:
            resource_violation = 'available_memory_reserve'
        if rc is not None:
            break
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds or resource_violation is not None:
            timed_out = elapsed > timeout_seconds
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=1.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            rc = child.returncode
            break
        time.sleep(0.01)
    return {
        'exit_status': rc,
        'timed_out': timed_out,
        'peak_rss_bytes': peak_rss,
        'peak_pss_bytes': peak_pss,
        'min_available_bytes': min_available,
        'max_swap_used_bytes': max_swap_used,
        'resource_violation': resource_violation,
        'wall_seconds': time.monotonic() - started,
    }


if __name__ == '__main__':
    import sys
    print(run_monitored([sys.executable, '-c', 'import time; time.sleep(.1)'], 2.0))
