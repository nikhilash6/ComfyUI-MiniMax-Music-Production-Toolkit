"""Tests for the model advisor (R03): hardware class, fit verdicts, star ratings.

What is pinned here:

* the advisor reads the *catalog*, so a model that can be downloaded is also a model
  that can be recommended - and an entry without a rating is reported as unrated
  instead of being given one;
* files that only work together (a Whisper checkpoint's five files, FlashSR's three
  weights) are one artifact, while files that are alternatives in the same folder
  (three MiniMax diffusion quantizations) stay separate;
* a fit verdict is computed from the *measured* free budget with a stated margin, and
  an unreadable device yields ``unknown`` - never a claim that something fits;
* a group is also checked as a *combination*: a diffusion model and its text encoder
  are loaded by the same run, so two files that each fit can still not fit together;
* nothing in this module downloads or changes a setting.
"""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _toolkit_bootstrap import load_entry_point

GIB = 1024 ** 3


class AdvisorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package, _ = load_entry_point()
        cls.package = package
        cls.advisor = importlib.import_module(package.__name__ + ".model_advisor")
        cls.downloader = importlib.import_module(package.__name__ + ".model_downloader")
        cls.resources = importlib.import_module(package.__name__ + ".resource_profiles")
        cls.whisper = importlib.import_module(package.__name__ + ".whisper_lyrics")

    def snapshot(self, vram_gib=None, free_gib=None, ram_gib=32.0, ram_free_gib=20.0, kind="cpu"):
        devices = []
        if kind == "cuda":
            devices.append(self.resources.DeviceInfo(
                id="cuda:0", name="Test GPU", kind="cuda", backend="cuda", logical=0, physical=0,
                vram_total_bytes=int(vram_gib * GIB) if vram_gib else None,
                vram_free_bytes=int(free_gib * GIB) if free_gib else None,
            ))
        if kind == "mps":
            devices.append(self.resources.DeviceInfo(id="mps:0", name="Apple MPS", kind="mps", backend="mps"))
        return self.resources.ResourceSnapshot(
            cpu_count=12, ram_total_bytes=int(ram_gib * GIB), ram_available_bytes=int(ram_free_gib * GIB),
            devices=devices,
        )


class HardwareClassTests(AdvisorTestCase):
    def test_class_is_the_smallest_bucket_of_the_largest_card(self):
        self.assertEqual(self.resources.hardware_class(self.snapshot(12, 11, kind="cuda")), "vram_12")
        self.assertEqual(self.resources.hardware_class(self.snapshot(48, 40, kind="cuda")), "vram_32")
        self.assertEqual(self.resources.hardware_class(self.snapshot()), "cpu")

    def test_an_unreadable_card_is_not_promoted_to_a_number(self):
        # "unknown" and "nothing" are different answers: a device whose memory could
        # not be read must not be classified as a small card (or as CPU-only).
        self.assertEqual(self.resources.hardware_class(self.snapshot(vram_gib=None, kind="cuda")), "vram_unknown")

    def test_apple_silicon_uses_unified_memory(self):
        self.assertEqual(self.resources.hardware_class(self.snapshot(kind="mps", ram_gib=32)), "vram_32")


class ArtifactTests(AdvisorTestCase):
    def setUp(self):
        self.candidates = self.advisor.catalog_candidates()

    def test_every_catalog_file_has_a_rating_and_a_target(self):
        self.assertTrue(self.candidates)
        for artifact in self.candidates:
            self.assertIsNotNone(artifact["rating"], artifact["name"])
            self.assertGreaterEqual(artifact["rating"], 1)
            self.assertLessEqual(artifact["rating"], 5)
            self.assertEqual(len(artifact["stars"]), 5)
            self.assertTrue(artifact["target"], f"{artifact['name']} has no target to download into")
            self.assertTrue(artifact["files"])

    def test_a_whisper_checkpoint_is_one_artifact_of_five_files(self):
        whisper = [a for a in self.candidates if a["group"] == "whisper"]
        self.assertEqual(len(whisper), 3, "large-v3 plus the two turbo variants")
        for artifact in whisper:
            self.assertEqual(len(artifact["files"]), 5, artifact["files"])
            self.assertEqual(artifact["name"], Path(artifact["target"]).name)
            self.assertGreater(artifact["bytes"], 0)

    def test_alternative_quantizations_stay_separate_artifacts(self):
        # Three MiniMax diffusion files share one folder but are three models - the
        # packing rule must not merge them into a 16 GiB "artifact".
        diffusion = [a for a in self.candidates if a["group"] == "minimax" and a["role"] == "diffusion"]
        self.assertEqual(len(diffusion), 3)
        for artifact in diffusion:
            self.assertEqual(len(artifact["files"]), 1)

    def test_flashsr_reads_from_a_reachable_source(self):
        flashsr = [a for a in self.candidates if a["group"] == "flashsr"]
        self.assertEqual(len(flashsr), 1)
        self.assertEqual(len(flashsr[0]["files"]), 3)
        self.assertEqual(flashsr[0]["repo_id"],
                         "laion/FlashSR_One-step_Versatile_Audio_Super-resolution")

    def test_an_unrated_entry_is_reported_as_unrated(self):
        config = {"llm": {"directory": "models/llm", "files": [
            {"name": "x.gguf", "repo_id": "org/repo", "filename": "x.gguf", "bytes": 1024, "revision": "a" * 40},
        ]}}
        candidate = self.advisor.catalog_candidates(config)[0]
        self.assertIsNone(candidate["rating"])
        self.assertEqual(candidate["stars"], "☆☆☆☆☆")


class FitVerdictTests(AdvisorTestCase):
    def verdict(self, gib, machine):
        candidate = {"bytes": int(gib * GIB)}
        return self.advisor.fit_verdict(candidate, self.advisor.hardware_summary(machine))

    def test_a_model_with_room_to_spare_fits(self):
        result = self.verdict(2.0, self.snapshot(12, 12, kind="cuda"))
        self.assertEqual(result["verdict"], "ok")
        self.assertGreater(result["free_after_bytes"], 0)
        self.assertIn("margin", result["reason"])

    def test_weights_without_room_for_the_margin_are_tight(self):
        result = self.verdict(9.0, self.snapshot(12, 12, kind="cuda"))
        self.assertEqual(result["verdict"], "tight")
        self.assertGreater(result["missing_bytes"], 0)

    def test_a_model_larger_than_the_budget_is_too_large(self):
        result = self.verdict(20.0, self.snapshot(12, 12, kind="cuda"))
        self.assertEqual(result["verdict"], "too_large")

    def test_an_unreadable_budget_never_claims_a_fit(self):
        result = self.verdict(1.0, self.snapshot(vram_gib=None, kind="cuda"))
        self.assertEqual(result["verdict"], "unknown")
        self.assertIn("not a size test", result["reason"])

    def test_default_machine_without_any_detected_value_stays_unknown(self):
        machine = {"class": "cpu", "budget_bytes": None}
        result = self.advisor.fit_verdict({"bytes": 1024}, machine)
        self.assertEqual(result["verdict"], "unknown")

    def test_the_margin_is_two_gib_or_twenty_percent(self):
        small = self.advisor._needed_bytes(1 * GIB) - 1 * GIB
        large = self.advisor._needed_bytes(40 * GIB) - 40 * GIB
        self.assertEqual(small, 2 * GIB)
        self.assertEqual(large, int(40 * GIB * 0.20))


class CombinationTests(AdvisorTestCase):
    def test_two_files_that_each_fit_can_still_not_fit_together(self):
        machine = self.advisor.hardware_summary(self.snapshot(16, 16, kind="cuda"))
        pair = [{"name": "a", "bytes": 7 * GIB}, {"name": "b", "bytes": 7 * GIB}]
        for pick in pair:
            self.assertEqual(self.advisor.fit_verdict(pick, machine)["verdict"], "ok")
        result = self.advisor.combination_verdict(pair, machine)
        self.assertIn(result["verdict"], ("tight", "too_large"))
        self.assertEqual(result["bytes"], 14 * GIB)
        self.assertEqual(result["names"], ["a", "b"])

    def test_an_incomplete_selection_says_unknown(self):
        result = self.advisor.combination_verdict(
            [], self.advisor.hardware_summary(self.snapshot(16, 16, kind="cuda")))
        self.assertEqual(result["verdict"], "unknown")

    def test_recommendation_names_a_smaller_set_that_fits(self):
        report = self.advisor.recommend_models(
            resources=self.snapshot(12, 11, kind="cuda"),
            base_path=Path(tempfile.gettempdir()) / "_advisor_absent",
        )
        artwork = next(group for group in report["groups"] if group["group"] == "flux2")
        combination = artwork["combination"]
        self.assertIn(combination["verdict"], ("tight", "too_large"))
        suggestion = combination.get("suggestion")
        self.assertIsNotNone(suggestion, "a 12 GiB card must be offered the fp8/fp4 variant set")
        self.assertLess(suggestion["bytes"], combination["bytes"])
        self.assertTrue(any("fp8" in name for name in suggestion["names"]), suggestion["names"])


class RecommendationTests(AdvisorTestCase):
    def test_a_role_with_nothing_that_fits_says_so_instead_of_picking(self):
        report = self.advisor.recommend_models(
            resources=self.snapshot(4, 1.2, kind="cuda"),
            base_path=Path(tempfile.gettempdir()) / "_advisor_absent",
        )
        roles = {role["role"]: role for group in report["groups"] for role in group["roles"]}
        self.assertEqual(roles["chat"]["pick"], "", "no language model fits a 1.2 GiB budget")
        self.assertIn("no candidate in this role fits", roles["chat"]["note"])

    def test_every_role_of_the_shipped_catalog_is_reported(self):
        report = self.advisor.recommend_models(
            resources=self.snapshot(24, 23, kind="cuda"),
            base_path=Path(tempfile.gettempdir()) / "_advisor_absent",
        )
        self.assertEqual({group["group"] for group in report["groups"]},
                         {"llm", "minimax", "yue2", "sheetsage2", "whisper", "flux2", "flashsr"})
        for group in report["groups"]:
            for role in group["roles"]:
                self.assertTrue(role["pick"], f"{group['group']}/{role['role']} picked nothing on 23 GiB")
                self.assertTrue(role["candidates"], role["role"])

    def test_the_report_is_json_serializable_for_an_app_ui(self):
        report = self.advisor.recommend_models(
            resources=self.snapshot(16, 15, kind="cuda"),
            base_path=Path(tempfile.gettempdir()) / "_advisor_absent",
        )
        text = json.dumps(report)
        self.assertIn("music_model_advice_v1", text)
        self.assertEqual(json.loads(text)["machine"]["class"], "vram_16")

    def test_cpu_only_machines_get_a_warning_for_the_generation_groups(self):
        report = self.advisor.recommend_models(
            resources=self.snapshot(), base_path=Path(tempfile.gettempdir()) / "_advisor_absent",
        )
        notes = {group["group"]: group["machine_note"] for group in report["groups"]}
        self.assertIn("CPU", notes["minimax"])
        self.assertIn("CPU", notes["flux2"])
        self.assertEqual(notes["llm"], "", "the language model path is usable without an accelerator")

    def test_the_text_report_shows_stars_and_the_machine(self):
        report = self.advisor.recommend_models(
            resources=self.snapshot(16, 15, kind="cuda"),
            base_path=Path(tempfile.gettempdir()) / "_advisor_absent",
        )
        text = "\n".join(self.advisor.format_advisor_lines(report, detail="summary"))
        self.assertIn("★", text)
        self.assertIn("class vram_16", text)
        self.assertIn("nothing here downloads", text.lower())

    def test_an_installed_artifact_is_not_recommended_away_silently(self):
        # A small stand-in catalog: the shipped GGUFs are gigabytes, and a file only
        # counts as installed when its size matches the catalog entry.
        config = {
            "llm": {"directory": "models/llm", "files": [
                {"name": "small.gguf", "repo_id": "org/small", "filename": "small.gguf",
                 "revision": "a" * 40, "bytes": 16, "rating": 2, "optional": True},
                {"name": "better.gguf", "repo_id": "org/better", "filename": "better.gguf",
                 "revision": "b" * 40, "bytes": 32, "rating": 5, "optional": True},
            ]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            folder = base / "models" / "llm"
            folder.mkdir(parents=True)
            (folder / "small.gguf").write_bytes(b"x" * 16)
            report = self.advisor.recommend_models(
                resources=self.snapshot(32, 30, kind="cuda"), config=config, base_path=base)
        chat = report["groups"][0]["roles"][0]
        installed = [c for c in chat["candidates"] if c["installed"]]
        self.assertEqual([c["name"] for c in installed], ["small.gguf"])
        self.assertEqual(chat["pick"], "small.gguf", "an installed model is not skipped silently")
        self.assertIn("rated higher", chat["note"])


class WhisperAlternativeTests(AdvisorTestCase):
    def test_the_group_check_fetches_only_the_default_checkpoint(self):
        # The smaller turbo folders are alternatives: a checkbox must not download
        # three whisper checkpoints.
        entries = self.downloader.normalize_model_entries(
            self.downloader.load_models_config(), minimax=False, yue2=False, sheetsage2=False,
            flux2=False, flashsr=False, llm=False, whisper=True,
        )
        names = {entry["name"] for entry in entries}
        self.assertEqual(names, {"model.bin", "config.json", "preprocessor_config.json",
                                 "tokenizer.json", "vocabulary.json"})
        targets = {entry["target"] for entry in entries}
        self.assertEqual(targets, {"models/audio_encoders/whisper-large-v3"})

    def test_the_dropdown_offers_every_folder_the_catalog_can_provide(self):
        choices = self.whisper.whisper_model_choices()
        self.assertEqual(choices, ["whisper-large-v3", "whisper-large-v3-turbo",
                                   "whisper-large-v3-turbo-int8"])

    def test_an_unknown_folder_is_never_fetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "some-other-model"
            folder.mkdir()
            report = self.whisper.fetch_whisper_model("some-other-model", folder)
        self.assertEqual(report, [])

    def test_a_missing_catalog_checkpoint_is_fetched_when_it_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "whisper-large-v3-turbo"
            called = {}

            def fake_check(entries, base_path=None, auto_download=False):
                called["entries"] = list(entries)
                called["auto_download"] = auto_download
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "model.bin").write_bytes(b"weights")
                return [{"name": entry["name"], "target": str(folder / entry["name"]),
                         "status": "downloaded", "message": ""} for entry in entries]

            module = importlib.import_module(self.package.__name__ + ".model_downloader")
            folder = Path(tmp) / "whisper-large-v3-turbo"
            with patch.object(module, "check_file_entries", fake_check), \
                 patch.object(self.whisper, "_catalog_checkpoint_folder", lambda model: folder):
                report = self.whisper.fetch_whisper_model("whisper-large-v3-turbo", folder)
        self.assertTrue(called["auto_download"], "the selected model is the one that gets fetched")
        self.assertEqual(len(called["entries"]), 5)
        self.assertEqual({entry["target"] for entry in called["entries"]},
                         {"models/audio_encoders/whisper-large-v3-turbo"})
        self.assertTrue(report)

    def test_a_ready_checkpoint_is_not_fetched_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "whisper-large-v3"
            folder.mkdir()
            (folder / "model.bin").write_bytes(b"weights")
            module = importlib.import_module(self.package.__name__ + ".model_downloader")
            with patch.object(module, "check_file_entries", side_effect=AssertionError("must not fetch")):
                self.assertEqual(self.whisper.fetch_whisper_model("whisper-large-v3", folder), [])


class CatalogValidationTests(AdvisorTestCase):
    def test_a_rating_must_be_a_star_count(self):
        for bad in (0, 6, "four", 3.5, True):
            self.assertIsNotNone(self.downloader.validate_model_entry({"name": "x", "rating": bad}))
        self.assertIsNone(self.downloader.validate_model_entry({"name": "x", "rating": 4}))

    def test_hardware_classes_are_checked_against_the_detector(self):
        problem = self.downloader.validate_model_entry({"name": "x", "suits": ["vram_7"]})
        self.assertIsNotNone(problem)
        self.assertIn("vram_7", problem)
        self.assertIsNone(self.downloader.validate_model_entry({"name": "x", "suits": ["cpu", "vram_12"]}))

    def test_stars_are_rendered_the_same_way_everywhere(self):
        self.assertEqual(self.downloader.stars(4), "★★★★☆")
        self.assertEqual(self.downloader.stars(None), "☆☆☆☆☆")
        self.assertEqual(self.advisor.stars_row(2), "★★☆☆☆")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
