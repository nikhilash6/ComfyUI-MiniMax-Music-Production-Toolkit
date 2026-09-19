"""ComfyUI progress helpers: the node's bar and the tqdm bar ComfyUI itself draws.

Two different jobs, deliberately separate:

* :func:`make_progress_bar` returns the standard ``comfy.utils.ProgressBar`` - the bar
  rendered *inside* the node, the same one every ComfyUI sampler uses.
* :func:`track` returns the **tqdm bar ComfyUI's own nodes use** - the one YuE2 draws with
  ``comfy.utils.model_trange(..., desc=..., unit="token")``. It shows percentage, bar,
  count, elapsed time, remaining time and the rate, and it updates in place, so a long
  stage is visible without writing a line per step into the log.

The rate is a measurement, not an estimate: tqdm computes it from the updates it actually
received.
"""
from __future__ import annotations

import os
from typing import Any, Optional

#: ``MINIMAX_MUSIC_TOOLKIT_PROGRESS=off`` turns the bars off (the closing summary lines are
#: still written - those are part of the record, not decoration).
PROGRESS_ENV = "MINIMAX_MUSIC_TOOLKIT_PROGRESS"
_OFF_VALUES = {"0", "off", "false", "no", "none", "quiet", "silent"}

#: A rate is only reported once this much time has passed: an average over a few
#: milliseconds says nothing about the run, and a spike at the start helps nobody.
RATE_MIN_SECONDS = 0.5


def progress_enabled() -> bool:
    """Whether progress bars may be drawn at all (env switch, default on)."""
    raw = str(os.getenv(PROGRESS_ENV, "") or "").strip().lower()
    return raw not in _OFF_VALUES


try:  # pragma: no cover - depends on the host
    from comfy.utils import ProgressBar as _ComfyProgressBar  # type: ignore
except Exception:  # pragma: no cover - outside ComfyUI
    _ComfyProgressBar = None

try:  # pragma: no cover - depends on the host
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is a ComfyUI dependency, but stay safe
    _tqdm = None


class _NoopBar:
    """Stands in for a bar when tqdm is missing: same two methods, nothing drawn."""

    def update(self, amount: int = 1) -> None:
        pass

    def close(self) -> None:
        pass


def make_progress_bar(total: int) -> Any:
    """Return a ComfyUI ``ProgressBar`` for ``total`` steps, or a silent no-op."""
    if _ComfyProgressBar is not None:
        return _ComfyProgressBar(int(total))

    class _NoopProgress:
        def update(self, amount: int = 1) -> None:
            pass

        def update_absolute(self, value: int) -> None:
            pass

    return _NoopProgress()


def track(total: Optional[int] = None, desc: str = "", unit: str = "it",
          disable: Optional[bool] = None, **kwargs: Any) -> Any:
    """The tqdm bar ComfyUI's own nodes draw, for a stage that counts its own steps.

    ``unit`` is what the rate is reported in (``"token"``, ``"chunk"``, ...), so the bar
    reads like the YuE2 one: ``LLM streaming:  8%|█▏  | 1958/24576 [01:15<14:24, 26.1token/s]``.

    The bar writes to stderr, which is where ComfyUI's bars go - the toolkit's log lines
    go to stdout, so the two never overwrite each other. Outside ComfyUI (tests, scripts)
    or with ``MINIMAX_MUSIC_TOOLKIT_PROGRESS=off`` the bar still accepts the same calls but
    draws nothing, so callers never need a branch.
    """
    if disable is None:
        disable = not progress_enabled()
    if _tqdm is None:
        return _NoopBar()
    return _tqdm(total=total, desc=desc, unit=unit, disable=bool(disable), **kwargs)


def format_duration(seconds: float) -> str:
    """``83.4`` -> ``"1:23"`` (hours only when they occur)."""
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_rate(done: int, seconds: float, unit: str = "step",
                minimum_seconds: float = RATE_MIN_SECONDS) -> str:
    """A rate as ``"12.3 chunks/s"``, or an empty string when it would say nothing.

    Used for the closing summary line of a stage. The rate is the average since the stage
    started - the number the caller can defend from the values it actually observed - and
    it is withheld for the first :data:`RATE_MIN_SECONDS`, where an average is noise.
    """
    if done <= 0 or seconds < max(0.0, float(minimum_seconds)):
        return ""
    return f"{done / max(1e-3, seconds):.1f} {unit}/s"
