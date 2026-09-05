"""A Workload defines *what gets sent to the engine and how it's timed* --
kept separate from Technique so several techniques can share the same
request pattern (e.g. every prefix-reuse method needs a cold/warm pair)
without copy-pasting the request logic into every technique file.

Add new workloads here as your evaluation needs grow -- e.g. a future
`NeedleInHaystack(Workload)` that buries a fact in the context and scores
retrieval accuracy would slot in exactly like SyntheticFiller/ColdWarm do,
and its score would flow into ResultEntry.extra via Technique.extra_metrics().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "


@dataclass
class WorkloadOutput:
    prompt_tokens: int
    generated_tokens: int
    ttft_sec: float
    decode_time_sec: float
    total_time_sec: float
    # Cold-run metrics -- populated only by ColdWarm; zero for SyntheticFiller.
    # Having them on the same type means ResultEntry.cold_ttft_sec is always
    # present (just 0.0 for single-shot workloads), so CSV columns are stable.
    cold_ttft_sec: float = 0.0
    cold_total_sec: float = 0.0


class Workload(ABC):
    @abstractmethod
    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        raise NotImplementedError


def _make_prompt(context_length: int) -> str:
    return FILLER_SENTENCE * max(1, context_length // 9)


def _timed_generate(llm, prompt: str, max_tokens: int) -> WorkloadOutput:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    start_wall = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end_wall = time.perf_counter()

    out = outputs[0]
    prompt_tokens = len(out.prompt_token_ids)
    gen_tokens = len(out.outputs[0].token_ids)

    metrics = getattr(out, "metrics", None)
    has_engine_metrics = metrics and all(
        getattr(metrics, f, None) for f in ("first_token_time", "arrival_time", "finished_time")
    )
    if has_engine_metrics:
        ttft = metrics.first_token_time - metrics.arrival_time
        decode = metrics.finished_time - metrics.first_token_time
        total = metrics.finished_time - metrics.arrival_time
    else:
        total = end_wall - start_wall
        ttft = total
        decode = 0.0

    return WorkloadOutput(
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        ttft_sec=round(ttft, 4),
        decode_time_sec=round(decode, 4),
        total_time_sec=round(total, 4),
    )


class SyntheticFiller(Workload):
    """One cold prompt of the target length, one short generation. This is
    what baseline/flexgen-style techniques need -- pure memory/latency
    measurement, no reuse involved.

    cold_ttft_sec and cold_total_sec in the output will be 0.0 for this
    workload since there is no separate cold vs. warm distinction.
    """

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        return _timed_generate(llm, _make_prompt(context_length), max_tokens)


class ColdWarm(Workload):
    """Two-phase prefix-reuse workload for chat-style scenarios: a 'cold'
    request that populates the KV cache (simulating the first message in a
    conversation or a shared system prompt), then a 'warm' request reusing
    that prefix with a new instruction (simulating subsequent turns).

    This models the SME/team chat use-case: 10-15 people sharing context
    (e.g. a long document or thread) where repeated queries all reuse the
    same prefix. A warm TTFT that is meaningfully lower than cold TTFT is
    direct evidence that prefix reuse is working.

    Both cold and warm timings are returned so the harness can surface the
    delta as a first-class reported field rather than a derived annotation.
    """

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        prefix = _make_prompt(context_length)
        cold_prompt = prefix + "\n\nInstruction: Summarize."
        warm_prompt = prefix + "\n\nInstruction: What is the theme?"

        cold_out = _timed_generate(llm, cold_prompt, max_tokens)
        warm_out = _timed_generate(llm, warm_prompt, max_tokens)

        # Surface cold timing on the warm result so the harness gets both in
        # one return value and can compute the delta without a second output type.
        warm_out.cold_ttft_sec = cold_out.ttft_sec
        warm_out.cold_total_sec = cold_out.total_time_sec
        return warm_out
