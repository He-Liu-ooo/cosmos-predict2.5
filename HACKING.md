# HACKING

## Run
```
# launch docker
./run-docker.sh

# enter virtual env
source .venv/bin/activate

# run
./run.sh output_subdir_name
```

## Verbose

Check the container mounts:

```bash
docker inspect --format '{{json .Mounts}}' cosmos-predict25-container | jq .
```

Typical output shows two mounts:
- a bind mount of the repository to `/workspace`
- a Docker volume mounted at `/workspace/.venv`

Because the `.venv` directory is a named Docker volume, it hides the host's `.venv` at the same path inside the container. That is why editing the host `.venv` does not affect the running container.

Recommended, minimal solution (no container reconfiguration): install the package in editable mode inside the container's venv. This creates an egg-link in the venv pointing to the workspace source so edits to the host workspace take effect inside the container.

Commands to run inside the container (use `docker exec -it <container> bash` first):

```bash
# activate the container venv
source /workspace/.venv/bin/activate

# bootstrap pip if needed, then upgrade
/workspace/.venv/bin/python -m ensurepip --upgrade || true
/workspace/.venv/bin/python -m pip install --upgrade pip setuptools wheel

# install editable package
/workspace/.venv/bin/python -m pip uninstall -y cosmos-oss cosmos_oss || true
/workspace/.venv/bin/python -m pip install -e /workspace/packages/cosmos-oss

# verify
/workspace/.venv/bin/python -c "import cosmos_oss; print('cosmos_oss.__file__=', getattr(cosmos_oss,'__file__',None))"
```

After this, editing `/workspace/packages/cosmos-oss/cosmos_oss` from VS Code will be reflected when the container imports the package (restart the process or use `importlib.reload` in interactive sessions).

Optional: if you want `debug.log` to be INFO by default, change `_init_log_files` in `packages/cosmos-oss/cosmos_oss/__init__.py` to register the file sinks at `INFO` (only enable DEBUG when VERBOSE).

Example change:

```py
def _init_log_files(output_dir: Path):
    console_path = output_dir / "console.log"
    debug_path = output_dir / "debug.log"
    log.info(f"Log saved to {console_path}")
    log.logger.add(console_path, mode="w", level="INFO", format=_LOGGER_FORMAT, filter=_console_filter, enqueue=True, catch=False)
    log.logger.add(debug_path, mode="w", level="INFO", format=_LOGGER_FORMAT, filter=log._rank0_only_filter, enqueue=True, catch=False)
```

This keeps debug-level output disabled by default; enable DEBUG only when you explicitly request verbose logging.

## Configuration
Currently there are 3 action chunks and 35 denoising steps

## Profiling
### pyinstrument
File: `cosmos_predict2/_src/predict2/conditioner.py`

Configuration snippet:

```python
profile: bool = True
```

To render a saved pyinstrument session as HTML:

```bash
pyinstrument --load profile.pyisession -r html -o report.html
```

### torch.profiler
- Modify profiling-related params in `assets/action_conditioned/basic/inference_params.json`.
- To add or change defaults, edit `cosmos_predict2/action_conditioned_config.py`.

Example trace path (existing):

```
outputs/action_conditioned/basic/profiling/torch_trace/action_chunk_iteration_3/rank0_trace.json.gz
```

Steps to view a torch.profiler trace:

1. run `unzip-trace.sh` to unzip a trace if neccessary
2. run `tensorboard.sh` to unzip a browser


### torch.cuda.Event

Use lightweight GPU events to collect accurate, low-overhead timings without repeatedly blocking the device. Key recommendations:

- Lightweight events: Only create `torch.cuda.Event` start/end pairs on the GPU. Do not call `torch.cuda.synchronize()` or compute elapsed times at every `stop()` — those calls block the GPU and distort latency measurements.

- Buffer metadata: Record start/end events together with metadata (for example: `action_chunk`, `denoise_step`/`step`, `block`, `layer`, `rank`) into an in-memory buffer instead of writing immediately.

- Attach via hooks: Register instrumentation hooks at the outer inference driver and forward timers into deep submodules. This lets you register timers once per sampling session and avoids repeated hook setup.

- Batch synchronization and write: At well-defined safe points (for example, the end of each action chunk, after every N timers, or at process exit) call `torch.cuda.synchronize()` once, compute elapsed times for all paired events in the buffer, then write the numeric results to disk in bulk (JSONL or CSV). Batching amortizes the synchronization cost and prevents repeated GPU stalls.

- Non-blocking IO: Perform disk writes from a background thread or asynchronously (or at minimum, append in reasonably sized batches). Write per-rank files to avoid concurrent-write conflicts and merge them offline when needed.

Example sketch (conceptual, not production-ready):

```py
# buffer events and metadata
event_buffer = []  # list of (start_event, end_event, metadata)

# when instrumenting:
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
# ... run op ...
end.record()
event_buffer.append((start, end, metadata_dict))

# at safe point:
torch.cuda.synchronize()
results = []
for s, e, meta in event_buffer:
    ms = s.elapsed_time(e)
    meta['elapsed_ms'] = float(ms)
    results.append(meta)

# write results in bulk (JSONL/CSV) from a background thread
```

Files and locations updated for this instrumentation:

```
cosmos_predict2/_src/imaginaire/utils/profiling.py
cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py
cosmos_predict2/action_conditioned.py
```

How to enable: set `enable_torch_cuda_event` to `True` in your `inference_params.json`.
How to plot: use `scripts/plot_torch_cuda_event_breakdown.py`.

## CUDA Graph
CUDA Graph support is experimental and not fully implemented. Do not enable `enable_cuda_graph` in `assets/action_conditioned/basic/inference_params.json` — setting it to `true` will likely cause the model to fail at runtime.