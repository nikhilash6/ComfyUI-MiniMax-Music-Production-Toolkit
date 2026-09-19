"""The FlashSR refinement stage must never end a run it cannot serve (F08).

The stage is a quality step with a pass-through fallback, so an unavailable model -
no network, a source that stopped answering, a full disk, the download switched off -
has to be *reported and skipped*, not raised: a run that can still produce the song must
not die because a refinement is missing.

Pinned here:

* the availability probe reports a reason instead of raising, and never treats an
  unreadable state as "ready";
* the audio node passes its input through unchanged, in the same AUDIO form, and says
  in its settings JSON that it was skipped and why;
* the model check treats a failed *FlashSR* download as a warning (the stage will skip
  itself), while a failed song-model download stays fatal.
"""
from __future__ import annotations

import importlib
import json
import unittest

import _toolkit_bootstrap

_PACKAGE, _HOST = _toolkit_bootstrap.load_entry_point()
flashsr = importlib.import_module(f"{_PACKAGE.__name__}.flashsr_audio")
autodownload = importlib.import_module(f"{_PACKAGE.__name__}.minimax_autodownload")


def audio_payload(samples=16000, rate=44100):
    import torch

    return {"waveform": torch.zeros((1, 1, samples)), "sample_rate": rate}


class StubPatch:
    """Replace one module attribute for the duration of a test."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self.value

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


class WeightsStatusTests(unittest.TestCase):
    def status(self, report, auto_download=True):
        with StubPatch(flashsr, "check_file_entries",
                       lambda entries, base_path=None, auto_download=False: report):
            return flashsr.flashsr_weights_status(auto_download)

    def test_installed_weights_are_ready(self):
        status = self.status([
            {"name": "student_ldm.pth", "status": "present", "target": "x", "message": ""},
            {"name": "sr_vocoder.pth", "status": "present", "target": "x", "message": ""},
            {"name": "vae.pth", "status": "present", "target": "x", "message": ""},
        ])
        self.assertTrue(status["ready"])
        self.assertEqual(sorted(status["files"]),
                         ["sr_vocoder.pth", "student_ldm.pth", "vae.pth"])

    def test_an_unreachable_source_is_reported_not_raised(self):
        status = self.status([
            {"name": "student_ldm.pth", "status": "failed", "target": "x",
             "message": "HTTP 401: Unauthorized"},
        ])
        self.assertFalse(status["ready"])
        self.assertIn("HTTP 401", status["reason"])
        self.assertEqual(status["failed"], ["student_ldm.pth"])

    def test_switched_off_downloads_are_reported_with_the_reason(self):
        status = self.status([
            {"name": "vae.pth", "status": "missing", "target": "x", "message": ""},
        ], auto_download=False)
        self.assertFalse(status["ready"])
        self.assertIn("auto-download disabled", status["reason"])
        self.assertEqual(status["missing"], ["vae.pth"])

    def test_the_probe_never_raises_when_the_check_explodes(self):
        def boom(*args, **kwargs):
            raise OSError("no route to host")

        with StubPatch(flashsr, "check_file_entries", boom):
            status = flashsr.flashsr_weights_status(True)
        self.assertFalse(status["ready"])
        self.assertIn("no route to host", status["reason"])

    def test_the_raising_variant_still_names_the_reason(self):
        with StubPatch(flashsr, "check_file_entries",
                       lambda entries, base_path=None, auto_download=False: [
                           {"name": "vae.pth", "status": "failed", "target": "x",
                            "message": "HTTP 404: Not Found"}]):
            with self.assertRaises(RuntimeError) as caught:
                flashsr._ensure_flashsr_weights(auto_download=True)
        self.assertIn("HTTP 404", str(caught.exception))


class SkippedStageTests(unittest.TestCase):
    def run_node(self, status):
        node = flashsr.MiniMaxFlashSRAudio()
        audio = audio_payload()
        with StubPatch(flashsr, "flashsr_weights_status", lambda auto_download=False: status):
            out_audio, settings = node.upscale(audio, auto_download=True)
        return audio, out_audio, json.loads(settings)

    def test_an_unavailable_stage_passes_the_audio_through_unchanged(self):
        audio, out_audio, settings = self.run_node({
            "ready": False, "reason": "student_ldm.pth could not be downloaded (HTTP 401)",
            "directory": "/models/audio/flashsr", "missing": [], "failed": ["student_ldm.pth"],
        })
        self.assertIs(out_audio, audio, "the very same AUDIO object must come back")
        self.assertEqual(settings["status"], "skipped")
        self.assertIn("HTTP 401", settings["reason"])
        self.assertEqual(settings["failed"], ["student_ldm.pth"])
        self.assertIn("optional", settings["note"])

    def test_the_skip_is_reported_with_the_weights_directory_and_the_way_back(self):
        _audio, _out, settings = self.run_node({
            "ready": False, "reason": "vae.pth not installed and auto-download disabled",
            "directory": "/models/audio/flashsr", "missing": ["vae.pth"], "failed": [],
        })
        self.assertEqual(settings["missing"], ["vae.pth"])
        self.assertEqual(settings["weights_dir"], "/models/audio/flashsr")
        self.assertEqual(settings["schema"], "flashsr_settings_v1")

    def test_invalid_input_still_fails_loudly(self):
        # A skipped stage is not a licence to accept broken input: the caller asked for
        # upscaling, so "no valid AUDIO" stays an error.
        node = flashsr.MiniMaxFlashSRAudio()
        with StubPatch(flashsr, "flashsr_weights_status",
                       lambda auto_download=False: {"ready": False, "reason": "x", "directory": "y",
                                                    "missing": [], "failed": []}):
            with self.assertRaises(RuntimeError):
                node.upscale(None, auto_download=True)


class CheckNodePolicyTests(unittest.TestCase):
    def check(self, entries, auto_download=True):
        node = autodownload.MiniMaxModelAutodownload()
        preflight = {"entries": entries, "summary": {"ok": True}}
        with StubPatch(autodownload, "preflight_models", lambda *a, **kw: preflight), \
             StubPatch(autodownload, "normalize_model_entries", lambda *a, **kw: []), \
             StubPatch(autodownload, "format_check_report", lambda entries: "report"), \
             StubPatch(autodownload, "format_preflight_report", lambda pf: []):
            # ``load_models_config`` stays real: which files belong to the optional
            # refinement stage is exactly what this policy reads from the catalog.
            return node.check(auto_download=auto_download)

    def test_a_failed_flashsr_download_does_not_abort_the_run(self):
        result = self.check([
            {"name": "student_ldm.pth", "status": "failed", "target": "x", "message": "HTTP 401"},
            {"name": "minimax_music3_dit_fp16.safetensors", "status": "present", "target": "x", "message": ""},
        ])
        self.assertEqual(result["result"], ("report",))

    def test_a_failed_song_model_download_stays_fatal(self):
        with self.assertRaises(RuntimeError) as caught:
            self.check([
                {"name": "minimax_music3_dit_fp16.safetensors", "status": "failed", "target": "x",
                 "message": "HTTP 404"},
            ])
        self.assertIn("minimax_music3_dit_fp16.safetensors", str(caught.exception))

    def test_the_flashsr_names_come_from_the_catalog(self):
        names = autodownload.flashsr_weight_names()
        self.assertIn("student_ldm.pth", names)
        self.assertIn("sr_vocoder.pth", names)
        self.assertIn("vae.pth", names)
        self.assertNotIn("minimax_music3_dit_fp16.safetensors", names)


if __name__ == "__main__":
    unittest.main()
