from __future__ import annotations
import os,resource,time

def rss_bytes(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def available_ram_bytes():
 try:
  with open('/proc/meminfo') as f: vals={x.split(':',1)[0]:int(x.split()[1])*1024 for x in f if ':' in x}
  return vals.get('MemAvailable',0)
 except OSError: return 0
def numerical_threads(): return int(os.environ.get('OMP_NUM_THREADS','0') or 0)
def now_ns(): return time.perf_counter_ns()
