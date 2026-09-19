"""Unit tests for the progress helpers.

Two promises are pinned here:

* ``make_progress_bar`` hands the node's own bar back inside ComfyUI and a silent no-op
  outside it, so toolkit code can always call it;
* ``track`` is the **same bar ComfyUI's own nodes draw** - tqdm with a description, a unit
  and a rate (YuE2 does it with ``comfy.utils.model_trange(..., unit="token")``). It is the
  display the user asked for: one line that updates in place, not one log line per step.
  It also has to be *quiet* when asked (env switch) and to survive a ComfyUI without tqdm.
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_progress_utils():
    pkg_name = "_progress_utils_test"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ROOT)]
    sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.progress_utils", ROOT / "progress_utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{pkg_name}.progress_utils"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class DurationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_progress_utils()

    def test_short_durations_read_as_minutes_and_seconds(self):
        self.assertEqual(self.mod.format_duration(0), "0:00")
        self.assertEqual(self.mod.format_duration(9.6), "0:09")
        self.assertEqual(self.mod.format_duration(83.4), "1:23")

    def test_long_durations_gain_an_hour_field(self):
        self.assertEqual(self.mod.format_duration(3661), "1:01:01")

    def test_a_broken_value_does_not_break_the_line(self):
        self.assertEqual(self.mod.format_duration(None), "?")
        self.assertEqual(self.mod.format_duration("nonsense"), "?")


class RateTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_progress_utils()

    def test_a_rate_needs_both_numbers(self):
        self.assertEqual(self.mod.format_rate(0, 10, "chunks"), "")
        self.assertEqual(self.mod.format_rate(20, 4, "chunks"), "5.0 chunks/s")

    def test_a_rate_is_withheld_until_it_means_something(self):
        # An average over a few milliseconds is noise, not a speed.
        self.assertEqual(self.mod.format_rate(3, 0.02, "chunks"), "")
        self.assertEqual(self.mod.format_rate(3, 2.0, "chunks"), "1.5 chunks/s")


class TrackTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_progress_utils()

    def test_the_unit_labels_the_rate_the_way_yue2_labels_its(self):
        # A disabled tqdm keeps its defaults, so the unit has to be checked on a bar that
        # really draws: it is what turns the rate column into "token/s".
        out = io.StringIO()
        bar = self.mod.track(10, desc="LLM streaming", unit="token", file=out, mininterval=0.0)
        self.assertEqual(bar.total, 10)
        bar.update(1)
        bar.refresh()
        bar.close()
        text = out.getvalue()
        self.assertIn("LLM streaming", text)
        self.assertIn("/10", text)
        self.assertIn("token", text)

    def test_the_drawn_line_looks_like_the_comfyui_bars(self):
        """``desc:  30%|███  | 3/10 [00:00<00:00, 12.3chunk/s]`` - the requested shape."""
        out = io.StringIO()
        bar = self.mod.track(10, desc="FlashSR upscaling", unit="chunk", file=out,
                             mininterval=0.0, disable=False)
        bar.update(3)
        bar.refresh()
        bar.close()
        text = out.getvalue()
        self.assertIn("FlashSR upscaling", text)
        self.assertIn("3/10", text)
        self.assertIn("%|", text, text)
        self.assertIn("chunk", text)

    def test_the_environment_switch_turns_the_bar_off(self):
        out = io.StringIO()
        with patch.dict(os.environ, {self.mod.PROGRESS_ENV: "off"}):
            self.assertFalse(self.mod.progress_enabled())
            bar = self.mod.track(10, desc="LLM", unit="token", file=out)
            bar.update(5)
            bar.close()
        self.assertEqual(out.getvalue(), "", "a silenced bar must not write anything")

    def test_a_switch_value_that_is_not_an_off_word_keeps_it_on(self):
        for value in ("on", "1", "yes", "", "loud"):
            with patch.dict(os.environ, {self.mod.PROGRESS_ENV: value}):
                self.assertTrue(self.mod.progress_enabled(), value)

    def test_without_tqdm_the_calls_still_work(self):
        with patch.object(self.mod, "_tqdm", None):
            bar = self.mod.track(5, desc="LLM", unit="token")
            self.assertIsInstance(bar, self.mod._NoopBar)
            bar.update(2)
            bar.close()

    def test_the_node_bar_is_still_a_silent_no_op_outside_comfyui(self):
        bar = self.mod.make_progress_bar(5)
        bar.update(1)
        bar.update_absolute(5)


if __name__ == "__main__":
    unittest.main()
