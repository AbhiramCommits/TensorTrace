# Contributing to TensorTrace

Thanks for helping improve TensorTrace. The project values reproducible
measurements over impressive-looking numbers: every metric in a report must come
from a real run, and every hardware-dependent feature must degrade to an
explicit `unavailable` state.

## Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Or use the Makefile:

```bash
make setup
make lint
make test
```

The fast suite runs offline and without root:

```bash
pytest -q -m "not slow"
```

The full suite includes a real, tiny inference run:

```bash
pytest -q --cov=tensortrace --cov-report=term-missing
```

## Code conventions

- Target Python 3.10+ and keep type hints on public functions and methods.
- Keep parsers pure: take text or objects in, return plain data. Put file reads
  and subprocess calls in thin wrappers such as `read_stat` or `run_perf_stat`.
- Do not fake unavailable metrics. Return `None` plus a reason, and make reports
  show `unavailable`.
- Add a docstring to every public module, class, and function.
- Run `ruff check .` before committing. The configured rules are `E`, `F`, `I`,
  `UP`, and `B` with a 100-character line limit.
- Add or update tests for every parser, threshold, and recommendation rule.

## Adding a new backend profiler

1. Create `tensortrace/profilers/<backend>_profiler.py`.
2. Implement a function with the signature
   `profile_<backend>(config: RunConfig) -> RunResult`.
3. Apply `config.effective_threads()` to the backend's thread controls.
4. Run `config.warmup` warmup iterations, then `config.iterations` measured
   iterations with `time.perf_counter_ns()`.
5. Populate the shared schema:
   - `layers`: `LayerTiming` rows with mean/total latency, calls, output bytes,
     and memory where available.
   - `iteration_latencies_ms` and `latency` (use `latency_stats`).
   - `warnings` for anything that could not be captured.
   - `extra` for backend-specific metadata.
6. Add the backend to `profile_backend` in `profilers/runner.py`, and to
   `DEFAULT_BACKENDS` / CLI choices if it should be user-selectable.
7. Add parser tests using captured text or a small recorded trace, plus a
   `@pytest.mark.slow` integration test that calls `profile_run`.
8. Do not bypass `profile_run` in production paths: it is what surrounds the
   backend with the system snapshot, sampling thread, and perf counters.

## Adding a new recommendation rule

1. Add a pure measurement helper in `tensortrace/analysis/hotspots.py`. It must
   accept a `RunResult` (or a perf dict) and return plain data with the measured
   values and any threshold flags.
2. Add a rule in `generate_recommendations` in
   `tensortrace/analysis/recommend.py`.
3. Use a stable, uppercase `code`, a severity (`critical`, `warning`, or
   `info`), a concrete message, and an `evidence` string that includes the
   actual measured numbers. Never emit a rule without evidence.
4. Add a test in `tests/test_recommend.py` that proves the code fires on its
   trigger and does not fire on a clean synthetic result.
5. Add a test to `tests/test_report.py` if the rule changes report output.
6. Keep ordering deterministic. Recommendations are sorted by severity then
   code, never by iteration order or randomness.

## Submitting changes

- Keep commits focused and use the project's existing message style.
- Include the commands you ran and their exact output for any new measured
  figure.
- Do not commit `.onnx-cache/`, virtual environments, or test caches. Generated
  `reports/` artifacts are intentionally committed.
- If a change affects the public schema, bump `SCHEMA_VERSION` and document the
  migration in `CHANGELOG.md`.
