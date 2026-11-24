import sys

from experiments.lite008h.monitor import run_monitored


def test_monitor_captures_normal_exit_and_timeout_cleanup():
    ok = run_monitored([sys.executable, "-c", "import time; time.sleep(.02)"], 1.0)
    assert ok["exit_status"] == 0
    assert not ok["timed_out"]
    assert ok["peak_rss_bytes"] > 0

    timed = run_monitored([sys.executable, "-c", "import time; time.sleep(2)"], 0.05)
    assert timed["timed_out"]
    assert timed["exit_status"] is not None
