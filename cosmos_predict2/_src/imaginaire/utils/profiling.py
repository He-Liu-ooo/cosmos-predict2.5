# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import time
from typing import Optional, Callable

import torch

from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

# (qsh 2024-11-23)  credits
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/profiling.py

# the number of warmup steps before the active step in each profiling cycle
TORCH_TRACE_WARMUP = 1
# TORCH_TRACE_WARMUP = 3

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


@contextlib.contextmanager
def maybe_enable_profiling(config, *, global_step: int = 0):
    # get user defined profiler settings
    enable_profiling = config.trainer.profiling.enable_profiling
    profile_freq = config.trainer.profiling.profile_freq

    if enable_profiling:
        trace_dir = os.path.join(config.job.path_local, "torch_trace")
        if distributed.get_rank() == 0:
            os.makedirs(trace_dir, exist_ok=True)

        rank = distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            log.info(f"Dumping traces at step {prof.step_num}")
            begin = time.monotonic()
            if rank in config.trainer.profiling.target_ranks:
                prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json.gz")
            log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

        log.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = TORCH_TRACE_WARMUP, 1
        wait = profile_freq - (active + warmup)
        assert wait >= 0, "profile_freq must be greater than or equal to warmup + active"

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=config.trainer.profiling.record_shape,
            profile_memory=config.trainer.profiling.profile_memory,
            with_stack=config.trainer.profiling.with_stack,
            with_modules=config.trainer.profiling.with_modules,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None

@contextlib.contextmanager
def maybe_enable_profiling_inference(
    config,
    *,
    global_step: int = 0,
    trace_handler_kind: str = "chrome",
    trace_handler_override: Optional[Callable] = None,
):
    # get user defined profiler settings
    enable_torch_profiler = config.enable_torch_profiler
    profile_freq = config.profile_freq

    if enable_torch_profiler:
        trace_dir = os.path.join(config.profiling_trace_path, "torch_trace")
        if distributed.get_rank() == 0:
            os.makedirs(trace_dir, exist_ok=True)

        rank = distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "chrome_action_chunk_iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            log.info(f"Dumping traces at step {prof.step_num}")
            begin = time.monotonic()
            if rank in config.profiling_target_ranks:
                prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json.gz")
            log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")


        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = TORCH_TRACE_WARMUP, 1
        wait = profile_freq - (active + warmup)
        assert wait >= 0, "profile_freq must be greater than or equal to warmup + active"
        log.info(f"Profiling active (warmup={warmup}, active={active}, wait={wait}). Traces will be saved at {trace_dir}")

        # choose on_trace_ready: prefer explicit override, then built-in tensorboard handler, else chrome json dump
        if trace_handler_override is not None:
            on_trace_ready = trace_handler_override
        elif trace_handler_kind == "tensorboard":
            tb_logdir = os.path.join(trace_dir, "tensorboard_action_chunk_iteration")
            os.makedirs(tb_logdir, exist_ok=True)
            on_trace_ready = torch.profiler.tensorboard_trace_handler(tb_logdir)
        else:
            on_trace_ready = trace_handler

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=on_trace_ready,
            record_shapes=config.profiling_record_shape,
            profile_memory=config.profiling_profile_memory,
            with_stack=config.profiling_with_stack,
            with_modules=config.profiling_with_modules,
            with_flops=config.profiling_with_flops,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(config, *, global_step: int = 0):
    enable_snapshot = config.trainer.profiling.enable_memory_snapshot
    if enable_snapshot:
        if config.trainer.profiling.save_s3:
            snapshot_dir = "s3://rundir"
        else:
            snapshot_dir = os.path.join(config.job.path_local, "memory_snapshot")
            if distributed.get_rank() == 0:
                os.makedirs(snapshot_dir, exist_ok=True)

        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES)
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name)
                if not config.trainer.profiling.save_s3 and not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                log.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()

                if rank in config.trainer.profiling.target_ranks:
                    easy_io.dump(
                        torch.cuda.memory._snapshot(),
                        f"{curr_snapshot_dir}/rank{rank}_memory_snapshot.pickle",
                    )
                log.info(f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds")

        log.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, config.trainer.profiling.profile_freq)
        try:
            yield profiler
        except torch.cuda.OutOfMemoryError as e:
            profiler.step(exit_ctx=True)
    else:
        yield None


"""Lightweight GPU timing helper using torch.cuda.Event.

This module provides CudaTimerCollection which records start/end
`torch.cuda.Event` pairs and flushes elapsed times to a jsonl file in
batches. If `torch.cuda.Event.elapsed_time` is unavailable for a pair
(e.g. events not completed or other failure), the code falls back to a
CPU-based duration and the output record is annotated with
`"timing_source": "cpu_fallback"` so downstream analysis can
filter or treat those entries specially.

Usage example:
    from cosmos_predict2._src.imaginaire.utils.cuda_timer import CudaTimerCollection
    timer = CudaTimerCollection(rank=0)

    # inside critical region
    with timer.measure(action=0, step=1, block=2, layer='ffn'):
        out = ffn_layer(x)

    # periodic flush (e.g. at end of action chunk)
    timer.flush_to_file(f"outputs/timings_rank{timer.rank}.jsonl")

Notes:
- This implementation batch-synchronizes once during flush (when
  `batch_sync=True`) to avoid repeated device synchronization that would
  distort GPU timing.
- Records that used CPU fallback are annotated with
  `"timing_source": "cpu_fallback"`.
"""

import json, threading
from typing import Dict, Any, List, Tuple

class CudaTimerCollection:
    def __init__(self, rank: int = 0):
        self.rank = rank  # for distributed setting
        self._lock = threading.Lock()
        self._measures: List[Dict[str, Any]] = []  # buffer of {meta, start, end events, ts_cpu}
        self._pending_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event, Dict[str, Any]]] = []

    @contextlib.contextmanager # enable writing with timer.measure(...)
    def measure(self, **meta):
        """Context manager to time a GPU region.

        Example meta keys: action, step, block, layer, extra tags.
        """
        # create events on the current CUDA device
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        cpu_ts = time.time() # for backup or comparison, in case GPU elapse fail
        # record start on default stream (non-blocking)
        start.record()
        try:
            yield # execute the code block being wrapped
        finally:
            end.record()
            with self._lock:
                self._pending_pairs.append((start, end, {"cpu_ts": cpu_ts, **meta}))
                
    def _drain_pending(self):
        with self._lock:
            pairs = self._pending_pairs
            self._pending_pairs = []
        return pairs
    
    def flush_to_file(self, path: str, batch_sync: bool = True, use_file_lock: bool = False) -> int:
        """Compute elapsed times for pending pairs and append to `path` as jsonl.

        If `batch_sync=True` we call `torch.cuda.synchronize()` once to ensure
        all events finished; this reduces synchronization overhead compared to
        synchronizing per-event.

        Returns the number of records written.
        """
        pairs = self._drain_pending()
        if not pairs:
            return 0

        if batch_sync:
            # single synchronization to ensure events completed
            torch.cuda.synchronize()

        lines = []
        for start, end, meta in pairs:
            timing_source = "gpu"
            try:
                # elapsed_time returns milliseconds (float)
                ms = start.elapsed_time(end)
            except Exception:
                # Fallback to CPU-based timing; mark the record so downstream
                # analysis can treat it specially.
                ms = (time.time() - meta.get("cpu_ts", time.time())) * 1000.0
                timing_source = "cpu_fallback"

            record = {
                "rank": self.rank,
                "elapsed_ms": float(ms),
                "timing_source": timing_source,
                **meta,
            }
            lines.append(record)

        # Ensure parent directory exists
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except Exception:
                pass

        # Append as jsonlines (simple and robust for downstream merging).
        # Optionally use an advisory file lock to avoid interleaved writes
        # when multiple processes/threads attempt to write the same file.
        if use_file_lock:
            try:
                import fcntl

                with open(path, "a", encoding="utf-8") as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX)
                    except Exception:
                        # best-effort locking; continue if locking unavailable
                        pass
                    for rec in lines:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                    try:
                        fcntl.flock(f, fcntl.LOCK_UN)
                    except Exception:
                        pass
            except Exception:
                # If fcntl/import fails, fall back to plain append
                with open(path, "a", encoding="utf-8") as f:
                    for rec in lines:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with open(path, "a", encoding="utf-8") as f:
                for rec in lines:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return len(lines)

    def aggregate_stats(self, records: List[Dict[str, Any]]):
        from collections import defaultdict
        stats = defaultdict(list)
        for r in records:
            key = (r.get("action"), r.get("step"), r.get("block"), r.get("layer"))
            stats[key].append(r["elapsed_ms"])
        out = {}
        import numpy as np
        for k, v in stats.items():
            arr = np.array(v)
            out[k] = {
                "count": len(v),
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }
        return out

from contextvars import ContextVar
CURRENT_DENOISE_STEP: ContextVar[int | None] = ContextVar("CURRENT_DENOISE_STEP", default=None)
# Context var for the current action chunk id (set by the inference driver per chunk)
CURRENT_ACTION_CHUNK: ContextVar[int | None] = ContextVar("CURRENT_ACTION_CHUNK", default=None)

""" hooks for profiling forward pass """
def _register_timing_hooks(model: torch.nn.Module, timer: CudaTimerCollection, name_filter=None) -> List:
    """
    Register pre/post forward hooks on modules whose name matches name_filter.
    name_filter: callable(name) -> bool or regex-like check
    Returns list of hook handles for later removal.
    """
    handles = []

    def make_hooks(mod_name: str):
        # pre-hook: record start event on appropriate device/stream
        def pre_hook(module, input):
            # determine device from first tensor input fallback to module params
            dev = None
            if isinstance(input, (list, tuple)) and len(input) and isinstance(input[0], torch.Tensor):
                dev = input[0].device
            else:
                try:
                    dev = next(module.parameters()).device
                except StopIteration:
                    dev = torch.device("cuda")
            stream = torch.cuda.current_stream(device=dev)
            start = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            # put start and device on module to retrieve in post
            setattr(module, "_cuda_timer_start_pair", (start, dev))

        # post-hook: record end event and push pair + meta into timer's pending buffer
        def post_hook(module, input, output):
            pair = getattr(module, "_cuda_timer_start_pair", None)
            if pair is None:
                return
            start, dev = pair
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(device=dev))
            # assemble meta - include denoise step and current action chunk from contextvars
            step = CURRENT_DENOISE_STEP.get(None)
            action_chunk = CURRENT_ACTION_CHUNK.get(None)
            meta = {"action_chunk": action_chunk, "denoise_step": step, "layer_name": mod_name}
            # append to timer pending (use its lock or a method)
            with timer._lock:
                timer._pending_pairs.append((start, end, meta))

        return pre_hook, post_hook

    for name, sub in model.named_modules():
        if name_filter is None or (callable(name_filter) and name_filter(name)):
            pre, post = make_hooks(name)
            h1 = sub.register_forward_pre_hook(pre)
            h2 = sub.register_forward_hook(post)
            handles.extend([h1, h2])
    return handles

def unregister_hooks(handles: List):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass