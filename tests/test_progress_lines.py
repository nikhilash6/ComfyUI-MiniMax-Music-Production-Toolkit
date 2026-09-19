"""The streaming stages must show a bar - not a line per update.

FlashSR's chunk loop and the LLM's token stream both used to log a line every few percent,
which filled the console with near-identical lines. What is pinned here:

* both stages drive **one** tqdm bar (the one ComfyUI's own nodes draw) and update it once
  per chunk/token - so the bar advances and the rate in it is a real measurement;
* the toolkit's logger receives the start line and the closing summary, and nothing per
  chunk;
* a long stage reports the rate in its summary as well, so a log file that keeps only the
  logger's lines still says how fast the stage ran;
* ``MINIMAX_MUSIC_TOOLKIT_PROGRESS=off`` silences the bar and keeps the record.
"""
from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import torch

import _toolkit_bootstrap

_PACKAGE, _HOST = _toolkit_bootstrap.load_entry_point()
llm = importlib.import_module(f"{_PACKAGE.__name__}.llm_chat")
flashsr = importlib.import_module(f"{_PACKAGE.__name__}.flashsr_audio")
progress = importlib.import_module(f"{_PACKAGE.__name__}.progress_utils")


class RecordingBar:
    """Stands in for the tqdm bar: records what a caller advanced it by."""

    def __init__(self, total=None, desc="", unit="it", **kwargs):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.updates = []
        self.closed = False

    def update(self, amount=1):
        self.updates.append(int(amount))

    def close(self):
        self.closed = True

    @property
    def advanced(self):
        return sum(self.updates)


def lines_for(module, level="INFO"):
    """Capture the module's logger as a list of formatted records."""
    import contextlib
    import logging

    class Recorder(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    recorder = Recorder()
    module.LOGGER.addHandler(recorder)
    module.LOGGER.setLevel(getattr(logging, level))
    frame = contextlib.ExitStack()
    frame.callback(module.LOGGER.removeHandler, recorder)
    return recorder, frame


class FakeClock:
    """A monotonic clock that advances a fixed step on every call."""

    def __init__(self, step=1.0):
        self.now = 1000.0
        self.step = float(step)

    def monotonic(self):
        self.now += self.step
        return self.now


class FakeStream:
    """One ``delta`` per chunk, like llama.cpp streams (one piece per decoded token)."""

    def __init__(self, chunks=50, usage=None):
        self.chunks = int(chunks)
        self.usage = usage
        self.closed = False

    def __iter__(self):
        for index in range(self.chunks):
            yield {"choices": [{"delta": {"content": f"chunk {index} "}}]}
        if self.usage is not None:
            yield {"usage": self.usage, "choices": []}

    def close(self):
        self.closed = True


class FakeModel:
    def __init__(self, stream):
        self.stream = stream

    def create_chat_completion(self, **kwargs):
        return self.stream


def run_llm(chunks=50, usage=None, clock_step=1.0, expect_bar=True):
    """Run one streamed turn with a recording bar and a faked clock."""
    stream = FakeStream(chunks=chunks, usage=usage)
    bar = RecordingBar()
    recorder, frame = lines_for(llm)
    track_calls = []

    def fake_track(total=None, desc="", unit="it", **kwargs):
        track_calls.append({"total": total, "desc": desc, "unit": unit})
        bar.total, bar.desc, bar.unit = total, desc, unit
        return bar

    # llm_chat imports the helpers inside the function, so patching the module they come
    # from is the seam; the clock is faked globally because the caller imports `time`.
    clock = FakeClock(clock_step)
    with frame, patch.object(progress, "track", fake_track), \
            patch("time.monotonic", clock.monotonic):
        text, thinking, result_usage = llm._run_chat_streamed(FakeModel(stream), {"messages": []}, 24576)
    return bar, recorder.messages, result_usage, track_calls


class LlmProgressTests(unittest.TestCase):
    def test_one_bar_is_advanced_once_per_chunk(self):
        bar, messages, _usage, calls = run_llm(chunks=50)
        self.assertEqual(calls, [{"total": 24576, "desc": "LLM streaming", "unit": "token"}], calls)
        self.assertEqual(bar.advanced, 50)
        self.assertEqual(len(bar.updates), 50)
        self.assertTrue(bar.closed, "the bar is closed on every path")

    def test_the_log_gets_a_start_line_and_one_summary(self):
        _bar, messages, _usage, _calls = run_llm(chunks=50)
        joined = "\n".join(messages)
        self.assertNotIn("progress", joined.lower().replace("streaming", ""))
        self.assertEqual(sum(1 for line in messages if "generation started" in line), 1)
        finished = [line for line in messages if "streaming finished" in line]
        self.assertEqual(len(finished), 1, messages)
        self.assertIn("50 token(s)", finished[0])
        self.assertIn("in ", finished[0])

    def test_a_long_run_reports_the_rate_in_the_summary(self):
        # 50 tokens over a faked 50 seconds: the summary may name the average.
        _bar, messages, _usage, _calls = run_llm(chunks=50, clock_step=1.0)
        finished = next(line for line in messages if "streaming finished" in line)
        self.assertIn("token/s", finished)

    def test_the_backend_usage_is_the_authoritative_token_count(self):
        _bar, messages, usage, _calls = run_llm(
            chunks=40, usage={"prompt_tokens": 900, "completion_tokens": 38, "total_tokens": 938})
        self.assertEqual(usage["source"], "backend")
        token_lines = [line for line in messages if "token rate" in line]
        self.assertEqual(len(token_lines), 1, messages)
        self.assertIn("tok/s", token_lines[0])
        self.assertIn("counted by the backend", token_lines[0])

    def test_a_cancel_still_closes_the_bar(self):
        class Cancel(FakeStream):
            def __iter__(self):
                yield {"choices": [{"delta": {"content": "x"}}]}
                raise KeyboardInterrupt("cancelled")

        stream = Cancel()
        bar = RecordingBar()
        recorder, frame = lines_for(llm)
        with frame, patch.object(progress, "track", lambda *args, **kwargs: bar), \
                patch("time.monotonic", FakeClock().monotonic):
            with self.assertRaises(KeyboardInterrupt):
                llm._run_chat_streamed(FakeModel(stream), {"messages": []}, 8192)
        self.assertTrue(bar.closed)
        self.assertTrue(stream.closed)


class FakeRunnerModel:
    def __call__(self, x, lowpass_input=False):
        return torch.zeros((x.shape[0], flashsr.CHUNK_SAMPLES), dtype=torch.float32)


class FlashSrProgressTests(unittest.TestCase):
    def run_upscale(self, seconds=30, items=1, clock_step=1.0):
        audio = {
            "waveform": torch.zeros((int(items), 1, int(flashsr.REQ_SR * seconds)), dtype=torch.float32),
            "sample_rate": flashsr.REQ_SR,
        }
        saved_runner = flashsr._get_runner
        saved_weights = flashsr.flashsr_weights_status
        flashsr._get_runner = lambda _dir: {"model": FakeRunnerModel(), "device": "cpu"}
        flashsr.flashsr_weights_status = lambda auto_download=False: {
            "ready": True, "directory": "/nonexistent", "missing": [], "failed": [], "reason": "test"}
        bar = RecordingBar()
        calls = []

        def fake_track(total=None, desc="", unit="it", **kwargs):
            calls.append({"total": total, "desc": desc, "unit": unit})
            bar.total, bar.desc, bar.unit = total, desc, unit
            return bar

        recorder, frame = lines_for(flashsr)
        try:
            with frame, patch.object(flashsr, "track", fake_track), \
                    patch("time.monotonic", FakeClock(clock_step).monotonic):
                _audio, settings = flashsr.MiniMaxFlashSRAudio().upscale(audio, auto_download=False)
        finally:
            flashsr._get_runner = saved_runner
            flashsr.flashsr_weights_status = saved_weights
        return bar, recorder.messages, calls, settings

    def test_the_bar_counts_every_chunk_once(self):
        bar, messages, calls, _settings = self.run_upscale(seconds=30)
        self.assertEqual(calls, [{"total": 7, "desc": "FlashSR upscaling", "unit": "chunk"}], calls)
        self.assertEqual(bar.advanced, 7)
        self.assertEqual(len(bar.updates), 7)
        self.assertTrue(bar.closed)

    def test_no_line_per_chunk_reaches_the_log(self):
        _bar, messages, _calls, _settings = self.run_upscale(seconds=180, items=2)
        joined = "\n".join(messages)
        self.assertNotIn("FlashSR progress", joined)
        finished = [line for line in messages if "upscale finished" in line]
        self.assertEqual(len(finished), 1, messages)
        self.assertIn("78/78 chunks", finished[0])
        self.assertIn("2 item(s)", finished[0])
        self.assertIn("chunk/s", finished[0], "the summary names the rate")

    def test_a_silenced_bar_still_leaves_the_record(self):
        import io
        import os
        from unittest.mock import patch as mock_patch

        audio = {"waveform": torch.zeros((1, 1, int(flashsr.REQ_SR * 30))), "sample_rate": flashsr.REQ_SR}
        saved_runner = flashsr._get_runner
        saved_weights = flashsr.flashsr_weights_status
        flashsr._get_runner = lambda _dir: {"model": FakeRunnerModel(), "device": "cpu"}
        flashsr.flashsr_weights_status = lambda auto_download=False: {
            "ready": True, "directory": "/nonexistent", "missing": [], "failed": [], "reason": "test"}
        stream = io.StringIO()
        recorder, frame = lines_for(flashsr)
        try:
            with mock_patch.dict(os.environ, {progress.PROGRESS_ENV: "off"}), \
                    mock_patch("sys.stderr", stream), frame:
                flashsr.MiniMaxFlashSRAudio().upscale(audio, auto_download=False)
        finally:
            flashsr._get_runner = saved_runner
            flashsr.flashsr_weights_status = saved_weights
        self.assertEqual(stream.getvalue(), "", "a silenced bar draws nothing")
        self.assertTrue(any("upscale finished" in line for line in recorder.messages),
                        "the record still says what ran and how long it took")


if __name__ == "__main__":
    unittest.main()
