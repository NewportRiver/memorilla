# LITE-008H Resource Monitor Qualification

The monitor is implemented in `experiments/lite008h/monitor.py`.

It launches one child process in a new process group, redirects stdout and stderr to `DEVNULL` to prevent pipe backpressure, polls `/proc/<pid>/status` and `smaps_rollup`, enforces a wall timeout, sends SIGTERM to the process group, escalates to SIGKILL, waits for cleanup, and records exit status, peak RSS, peak PSS, timeout status, and wall time.

Qualification tests:

- Short-lived child: exited `0`, not timed out, peak RSS `10,711,040` bytes.
- Timeout child: terminated with SIGTERM (`-15`), timeout recorded true, peak RSS `10,833,920` bytes.

The previous H-equivalent instrumentation hang is not repeated. The root cause is supported by the prior command behavior: the native decoder emitted extensive diagnostics while its stderr pipe was captured without draining. No aggregate semantic RSS is reconstructed from that failed run.
