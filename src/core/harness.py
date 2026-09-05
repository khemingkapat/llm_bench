"""The unified runner. This is the part that used to be copy-pasted into
every benchmark_*.py file: reset memory, build the engine, run the workload,
measure, handle failures, clean up. Written once here; every technique just
plugs into it.

Optional --profile mode wraps the workload in torch.profiler, saves the
Chrome trace, and extracts a memcpy summary (bytes transferred CPU↔GPU and
peak achieved PCIe bandwidth) into ResultEntry.extra. This gives you the
same transfer accounting that the old profiler_trace_phase3 runs produced,
now integrated directly into the benchmark result JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .axis import axis_names
from .metrics import (
    ResultEntry,
    host_rss_mb,
    peak_gpu_memory_mb,
    reset_gpu_memory,
    throughput,
)
from .technique import Technique


@dataclass
class RunResult:
    model: str
    technique: str
    axes: list[str]
    results: list[ResultEntry]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Profiler trace parsing
# ---------------------------------------------------------------------------

def _parse_memcpy_summary(trace_path: str) -> dict:
    """Extract GPU↔CPU memory transfer stats from a Chrome trace file.

    Returns bytes_htod (CPU→GPU), bytes_dtoh (GPU→CPU), and peak bandwidth
    seen across all memcpy events. These come from the 'gpu_memcpy' category
    events that PyTorch profiler emits when profile_memory=True.
    """
    try:
        with open(trace_path) as f:
            trace = json.load(f)
    except Exception:
        return {}

    events = trace.get("traceEvents", [])
    htod_bytes = 0
    dtoh_bytes = 0
    peak_bw_gbs = 0.0

    for ev in events:
        if ev.get("cat") != "gpu_memcpy":
            continue
        args = ev.get("args", {})
        name = ev.get("name", "")
        b = args.get("bytes", 0)
        bw = args.get("memory bandwidth (GB/s)", 0.0)
        if "HtoD" in name:
            htod_bytes += b
        elif "DtoH" in name:
            dtoh_bytes += b
        if bw > peak_bw_gbs:
            peak_bw_gbs = bw

    return {
        "memcpy_htod_bytes": htod_bytes,
        "memcpy_dtoh_bytes": dtoh_bytes,
        "memcpy_peak_bw_gbs": round(peak_bw_gbs, 3),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(
    technique: Technique,
    model: str,
    contexts: list[int],
    max_tokens: int = 64,
    gpu_utilization: float = 0.85,
    profile: bool = False,
    profile_dir: str = "results/traces",
    dry_run: bool = False,
) -> RunResult:
    """Run a technique across all requested context lengths.

    Args:
        technique: The Technique instance to benchmark.
        model: HuggingFace model ID or local path.
        contexts: List of prompt token lengths to sweep.
        max_tokens: Tokens to generate per request.
        gpu_utilization: Fraction of VRAM vLLM may use for KV blocks.
        profile: If True, wrap the workload in torch.profiler and save a
            Chrome trace per context length. Memcpy stats are extracted and
            added to ResultEntry.extra automatically.
        profile_dir: Directory for trace files (created if absent).
        dry_run: If True, skip the actual LLM() call -- useful for verifying
            technique registration and routing without a GPU.
    """
    if not dry_run:
        from vllm import LLM  # noqa: F401 -- deferred so dry-run works without vllm

    technique.setup()
    workload = technique.workload()
    axes = axis_names(technique.axes)
    entries: list[ResultEntry] = []

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (dry-run)"
    print(f"=== {technique.name} :: axes={axes} ===")
    print(f"Model: {model} | GPU: {gpu_name}" + (" [DRY RUN]" if dry_run else ""))
    print("-" * 65)

    for ctx in contexts:
        print(f"\n[context length: {ctx}]")
        reset_gpu_memory()
        rss_before = host_rss_mb()
        llm = None

        try:
            if dry_run:
                # Emit a synthetic SUCCESS entry so callers get a parseable result.
                entry = ResultEntry(
                    requested_ctx=ctx,
                    status="DRY_RUN",
                    axes=axes,
                )
                print(f" -> DRY RUN OK (engine kwargs: {technique.engine_kwargs(ctx, max_tokens)})")
                entries.append(entry)
                continue

            from vllm import LLM

            llm = LLM(
                model=model,
                dtype="half",
                gpu_memory_utilization=gpu_utilization,
                max_model_len=ctx + max_tokens + 256,
                trust_remote_code=True,
                enforce_eager=True,
                **technique.engine_kwargs(ctx, max_tokens),
            )

            if profile:
                trace_path = _run_with_profiler(llm, workload, ctx, max_tokens, profile_dir, technique.name)
            else:
                trace_path = None

            output = workload.run(llm, ctx, max_tokens)
            rss_after = host_rss_mb()

            peak_alloc, peak_reserved = peak_gpu_memory_mb()
            tp = throughput(output.generated_tokens, output.decode_time_sec, output.total_time_sec)

            extra = technique.extra_metrics(llm, output)
            if trace_path:
                extra.update(_parse_memcpy_summary(trace_path))

            entry = ResultEntry(
                requested_ctx=ctx,
                status="SUCCESS",
                axes=axes,
                prompt_tokens=output.prompt_tokens,
                generated_tokens=output.generated_tokens,
                ttft_sec=output.ttft_sec,
                decode_time_sec=output.decode_time_sec,
                total_time_sec=output.total_time_sec,
                throughput_tok_per_sec=tp,
                cold_ttft_sec=output.cold_ttft_sec,
                cold_total_sec=output.cold_total_sec,
                peak_alloc_mb=peak_alloc,
                peak_reserved_mb=peak_reserved,
                host_rss_delta_mb=round(rss_after - rss_before, 2),
                extra=extra,
            )

            delta_label = (
                f" | cold TTFT {output.cold_ttft_sec:.3f}s → warm {output.ttft_sec:.3f}s"
                if output.cold_ttft_sec > 0
                else ""
            )
            print(
                f" -> peak VRAM {peak_alloc:.1f} MB | TTFT {output.ttft_sec:.3f}s | "
                f"{tp:.2f} tok/s | host RSS Δ {entry.host_rss_delta_mb:+.0f} MB"
                + delta_label
            )

        except Exception as e:  # noqa: BLE001 -- a failed context length shouldn't kill the sweep
            entry = ResultEntry(requested_ctx=ctx, status="FAILED", axes=axes, error=str(e))
            print(f" -> FAILED: {e}")

        finally:
            del llm
            reset_gpu_memory()

        entries.append(entry)

    return RunResult(
        model=model,
        technique=technique.name,
        axes=axes,
        results=entries,
    )


def _run_with_profiler(llm, workload, ctx: int, max_tokens: int, profile_dir: str, technique_name: str) -> str:
    """Wrap workload.run() in torch.profiler and return the saved trace path."""
    import torch.profiler as profiler

    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    trace_path = os.path.join(profile_dir, f"{technique_name}_ctx{ctx}.pt.trace.json")

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        workload.run(llm, ctx, max_tokens)

    prof.export_chrome_trace(trace_path)
    return trace_path
