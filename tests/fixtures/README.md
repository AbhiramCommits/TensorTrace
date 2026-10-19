"""Provenance of test fixtures.

- ``proc_stat.txt``, ``proc_status.txt``, ``meminfo.txt``: captured from a real
  Ubuntu 22.04 process inside the ``labbench/linux-host`` container
  (kernel 7.0.14-linuxkit, aarch64).
- ``perf_stat.csv``: real ``perf stat -x,`` output from perf 5.15.216 in a
  privileged Linux container. Hardware events were rejected by the virtualized
  PMU (``<not supported>``) while ``page-faults`` and ``context-switches`` were
  counted, which is exactly the mixed-output case the parser must handle.
- ``ort_profile.json``: real ONNX Runtime 1.31.0 profiling output from a tiny
  real model run on the build host.
- ``sysfs/node`` and ``sysfs/cache``: Linux-ABI-formatted topology samples. The
  Docker LinuxKit kernel used for capture exposes no NUMA node directories and
  omits some cache attributes (``size``, ``coherency_line_size``), so these
  fixtures supply representative 8-core / 2-node values. Parser behavior is
  tested against them; they are not claimed as measurements of the build host.
- ``cache_index0_*.txt`` / ``cache_index2_*.txt``: real captured cache
  ``level``/``type``/``shared_cpu_list`` files from the Linux container (the
  size and line-size files are absent there).
"""

from __future__ import annotations
