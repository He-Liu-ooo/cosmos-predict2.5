# HACKING
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

1. Decompress the trace (project includes `unzip-trace.sh` for convenience).
2. Download the decompressed trace file to your local machine (if the container cannot open it directly).
3. Open https://ui.perfetto.dev/ in your browser and load the trace file.
