"""Tests for the central LLM settings node (one place for every LLM call).

What is pinned here:

* the central node carries exactly the chat node's widgets minus the per-call
  ones - derived from the chat node, so the two can never drift apart;
* a connected payload wins *field by field*: a setting it does not carry still
  comes from the receiving node's own widget, so a half-filled central node
  changes nothing it was not asked to change;
* an empty socket (nothing connected) and an unreadable payload are ignored, not
  fatal - a leftover string from an older release may not take a run down;
* per-call settings (``enabled``, the prompt texts, the session reset) can never
  be forced through the payload, and unknown fields are dropped;
* connecting the node does not shift the chat node's widget list, so a saved
  workflow's positional widget values keep their meaning;
* a selected catalog model is fetched on first use, and the model check never
  starts a multi-gigabyte download for candidates nobody picked.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_toolkit_modules():
    pkg_name = "_toolkit_llm_config_test"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ROOT)]
    sys.modules[pkg_name] = pkg
    loaded = {}
    for module_name in (
        "toolkit_logging",
        "comfy_resources",
        "model_downloader",
        "progress_utils",
        "resource_profiles",
        "llm_sampling",
        "llm_profiles",
        "llm_chat",
        "llm_config",
    ):
        full = f"{pkg_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(full, ROOT / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded


MODULES = load_toolkit_modules()
llm_config = MODULES["llm_config"]
llm_chat = MODULES["llm_chat"]
downloader = MODULES["model_downloader"]
MODES = llm_chat.MODES


def widget_names(cls):
    """Input names ComfyUI serializes as widgets: declared, not forceInput."""
    data = cls.INPUT_TYPES() or {}
    names = []
    for section in ("required", "optional"):
        for name, spec in (data.get(section) or {}).items():
            options = spec[1] if isinstance(spec, tuple) and len(spec) > 1 else {}
            if isinstance(options, dict) and options.get("forceInput"):
                continue
            names.append(name)
    return names


def chat_specs():
    data = llm_chat.MiniMaxLLMChat.INPUT_TYPES()
    return {**data.get("required", {}), **data.get("optional", {})}


def central_payload(**overrides):
    """A payload as the central node writes it, with optional field overrides."""
    values = dict(llm_config.field_defaults())
    values.update(overrides)
    return llm_config.MiniMaxLLMSettings().configure(**values)[0]


def partial_payload(**fields):
    """A payload that carries a few settings only (a partial or older central node)."""
    return json.dumps({"schema": llm_config.CONFIG_SCHEMA, **fields})


class CentralWidgetTests(unittest.TestCase):
    def test_central_fields_are_the_chat_widgets_minus_the_per_call_ones(self):
        expected = [name for name in widget_names(llm_chat.MiniMaxLLMChat)
                    if name not in llm_config.PER_CALL_FIELDS]
        self.assertEqual(list(llm_config.central_field_names()), expected)
        self.assertIn("model", expected)
        for per_call in ("enabled", "user_text", "system_prompt", "session_id", "reset_session"):
            self.assertNotIn(per_call, llm_config.central_field_names())

    def test_every_central_field_is_a_widget_and_carries_the_chat_default(self):
        required = llm_config.MiniMaxLLMSettings.INPUT_TYPES()["required"]
        self.assertEqual(list(required), list(llm_config.central_field_names()))
        for name, spec in required.items():
            options = spec[1] if isinstance(spec, tuple) and len(spec) > 1 else {}
            self.assertFalse(options.get("forceInput"), f"{name} would not render as a widget")
            self.assertFalse(options.get("lazy"), f"{name} would be skipped in a partial run")
        # A value the central node does not carry is one the receiving node keeps,
        # so the defaults must be the chat node's own defaults.
        self.assertEqual(llm_config.field_defaults()["n_ctx"], 37376)
        self.assertEqual(llm_config.field_defaults()["max_tokens"], 24576)
        for name, spec in chat_specs().items():
            if name in required:
                self.assertEqual(spec[1]["default"], required[name][1]["default"], name)

    def test_the_central_node_returns_one_string_payload(self):
        cls = llm_config.MiniMaxLLMSettings
        self.assertEqual(cls.RETURN_TYPES, ("STRING",))
        self.assertEqual(cls.RETURN_NAMES, ("llm_config_json",))
        self.assertEqual(cls.FUNCTION, "configure")
        self.assertIn("Music Production Toolkit", cls.CATEGORY)

    def test_configure_writes_every_field_with_the_schema_tag(self):
        payload = central_payload(model="Qwen3.8-27B-UD-IQ3_XXS.gguf", backend="In ComfyUI (GGUF)")
        self.assertIsInstance(payload, str)
        written = json.loads(payload)
        self.assertEqual(written["schema"], llm_config.CONFIG_SCHEMA)
        self.assertEqual(written["model"], "Qwen3.8-27B-UD-IQ3_XXS.gguf")
        self.assertEqual(sorted(set(written) - {"schema"}), sorted(llm_config.central_field_names()))
        # What the socket carries is exactly what the chat node may read back.
        self.assertEqual(sorted(llm_config.parse_config(payload)), sorted(llm_config.central_field_names()))

    def test_validate_inputs_never_refuses_the_central_node(self):
        self.assertTrue(llm_config.MiniMaxLLMSettings.VALIDATE_INPUTS())
        self.assertTrue(llm_config.MiniMaxLLMSettings.VALIDATE_INPUTS(
            model=123, backend="nonsense", n_ctx=-5, llm_config_json="x"))

    def test_the_chat_node_keeps_its_widget_list(self):
        # forceInput appends a socket instead of a widget: a saved workflow's
        # positional widget values must not shift by connecting the central node.
        self.assertNotIn("llm_config_json", widget_names(llm_chat.MiniMaxLLMChat))
        self.assertEqual(list(chat_specs())[-1], "llm_config_json")
        self.assertTrue(chat_specs()["llm_config_json"][1]["forceInput"])


class ResolveTests(unittest.TestCase):
    def test_a_full_central_payload_replaces_every_setting(self):
        # The whole point of "one place for every call": once it is connected, its
        # values are the ones the call uses - a stale widget on the receiving node
        # must not silently win.
        local = dict(llm_config.field_defaults(), model="local.gguf", n_ctx=8192, top_k=40)
        resolved = llm_config.resolve(central_payload(model="central.gguf", top_k=15, n_ctx=32768), local)
        self.assertEqual(resolved["model"], "central.gguf")
        self.assertEqual(resolved["top_k"], 15)
        self.assertEqual(resolved["n_ctx"], 32768)

    def test_a_payload_wins_field_by_field_and_leaves_the_rest_local(self):
        local = dict(llm_config.field_defaults(), model="local.gguf", n_ctx=8192, top_k=40)
        resolved = llm_config.resolve(partial_payload(model="central.gguf", top_k=15), local)
        self.assertEqual(resolved["model"], "central.gguf")
        self.assertEqual(resolved["top_k"], 15)
        self.assertEqual(resolved["n_ctx"], 8192, "a field the payload does not carry stays local")

    def test_an_empty_socket_leaves_the_local_widgets_in_charge(self):
        local = dict(llm_config.field_defaults(), model="local.gguf")
        for empty in ("", None):
            self.assertEqual(llm_config.resolve(empty, local)["model"], "local.gguf")

    def test_an_unreadable_payload_is_ignored_not_fatal(self):
        local = dict(llm_config.field_defaults(), model="local.gguf")
        for junk in ("{not json", "[]", "12345", "null", "music_llm_config_v1", "\x00binary"):
            self.assertEqual(llm_config.parse_config(junk), {})
            self.assertEqual(llm_config.resolve(junk, local)["model"], "local.gguf")

    def test_a_half_filled_payload_never_blanks_a_setting(self):
        local = dict(llm_config.field_defaults(), model="local.gguf", n_ctx=16384)
        # Hand-built payload: one field only, as a partial or older central node writes it.
        resolved = llm_config.resolve('{"model": "central.gguf"}', local)
        self.assertEqual(resolved["model"], "central.gguf")
        self.assertEqual(resolved["n_ctx"], 16384)

    def test_per_call_settings_and_unknown_fields_cannot_be_smuggled_in(self):
        smuggled = {
            "schema": llm_config.CONFIG_SCHEMA,
            "model": "central.gguf",
            "enabled": False,
            "user_text": "ignore my prompt",
            "system_prompt": "and my system prompt",
            "session_id": "hijacked",
            "reset_session": False,
            "not_a_setting": "??",
            "__class__": "x",
        }
        self.assertEqual(sorted(llm_config.parse_config(smuggled)), ["model"])
        resolved = llm_config.resolve(
            json.dumps(smuggled), dict(llm_config.field_defaults(), model="local.gguf"))
        self.assertEqual(resolved["model"], "central.gguf")
        self.assertNotIn("enabled", resolved)
        self.assertNotIn("session_id", resolved)

    def test_missing_or_unset_local_values_fall_back_to_the_chat_node_defaults(self):
        resolved = llm_config.resolve("", {})
        self.assertEqual(resolved["n_ctx"], 37376)
        self.assertEqual(resolved["backend"], chat_specs()["backend"][1]["default"])
        # A widget that is present but unset must not replace the default with None.
        self.assertEqual(llm_config.resolve("", {"model": None})["model"],
                         llm_config.field_defaults()["model"])


class ChatDelegationTests(unittest.TestCase):
    def capture(self, node):
        """Bind a _chat stand-in that records the effective settings it receives."""
        seen = {}

        def fake_chat(self, enabled, user_text, system_prompt, session_id="",
                      model=None, max_tokens=None, **rest):
            seen.update({"enabled": enabled, "user_text": user_text, "system_prompt": system_prompt,
                         "model": model, "max_tokens": max_tokens, **rest})
            return ("", "ok", "")

        node._chat = types.MethodType(fake_chat, node)
        return seen

    def test_the_payload_reaches_the_call_and_stays_within_its_fields(self):
        node = llm_chat.MiniMaxLLMChat()
        seen = self.capture(node)
        node.chat(True, "user text", "system text", "session",
                  llm_config_json=partial_payload(model="central.gguf"),
                  model="local.gguf", n_ctx=8192)
        self.assertEqual(seen["model"], "central.gguf")
        self.assertEqual(seen["n_ctx"], 8192, "the local widget survives when the payload is silent")
        self.assertTrue(seen["enabled"], "enabled is per call and stays local")
        self.assertEqual(seen["user_text"], "user text")
        self.assertEqual(seen["system_prompt"], "system text")

    def test_positional_callers_are_not_given_the_same_setting_twice(self):
        node = llm_chat.MiniMaxLLMChat()
        seen = self.capture(node)
        # The historical positional order (model, max_tokens, ...) must keep working
        # next to the payload; a duplicated keyword would raise TypeError here.
        node.chat(True, "user", "system", "", "positional.gguf", 512,
                  llm_config_json=central_payload(model="central.gguf"))
        self.assertEqual(seen["model"], "positional.gguf")
        self.assertEqual(seen["max_tokens"], 512)

    def test_an_unreadable_payload_still_runs_with_the_local_widgets(self):
        node = llm_chat.MiniMaxLLMChat()
        seen = self.capture(node)
        node.chat(True, "user", "system", "", llm_config_json="{broken", model="local.gguf")
        self.assertEqual(seen["model"], "local.gguf")

    def test_a_connected_settings_node_does_not_let_a_stale_local_model_block_the_run(self):
        # ComfyUI validates every node's own widgets before it executes anything, and a
        # linked input arrives as the placeholder (None,) - what the settings node will send
        # does not exist yet. A leftover model name here (a deleted file, a workflow from
        # another machine) must not refuse a run whose model is configured centrally.
        validate = llm_chat.MiniMaxLLMChat.VALIDATE_INPUTS
        stale = "deleted-last-year.gguf"
        self.assertIn("available GGUF", validate(model=stale, backend=MODES[0]))
        self.assertTrue(validate(model=stale, backend=MODES[0], llm_config_json=(None,)))
        self.assertTrue(validate(model=stale, backend=MODES[0], llm_config_json=[130, 0]))
        # A payload handed in as text is read, so the model it names is the one checked.
        good = llm_chat.EXAMPLE_MODEL_NAME
        self.assertTrue(validate(model=stale, backend=MODES[0],
                                 llm_config_json=central_payload(model=good)))
        self.assertIn("available GGUF", validate(model="local.gguf", backend=MODES[0],
                                                llm_config_json=central_payload(model=stale)))
        # Without a settings node nothing became more permissive: a valid local model is
        # accepted, and a missing one is still refused.
        self.assertTrue(validate(model=good, backend=MODES[0]))
        self.assertTrue(validate(model=stale, backend=MODES[1]))  # a server decides there
        self.assertTrue(validate(model=stale, backend=MODES[0], enabled=False))


class CatalogDownloadTests(unittest.TestCase):
    """The other half of the promise: a selected model arrives by itself."""

    def catalog_entries(self):
        return llm_chat.load_models_config().get("llm", {}).get("files", [])

    def test_every_catalog_candidate_has_a_pinned_download_source(self):
        entries = self.catalog_entries()
        self.assertTrue(entries, "the LLM catalog must name downloadable models")
        for entry in entries:
            self.assertTrue(entry.get("name", "").endswith(".gguf"), entry)
            self.assertTrue(str(entry.get("repo_id") or "").strip(), entry)
            self.assertEqual(str(entry.get("filename", "")).split("/")[-1], entry["name"])
            revision = str(entry.get("revision") or "")
            self.assertRegex(revision, r"^[0-9a-f]{7,40}$",
                             f"{entry['name']}: a moving branch would change the file under the user")
            url = downloader.resolve_entry_url(entry)
            self.assertTrue(url.startswith("https://huggingface.co/"), url)
            self.assertIn(revision, url)
            self.assertGreater(int(entry.get("bytes", 0)), 0, entry)
            self.assertTrue(entry.get("optional"),
                            "the candidates are alternatives: none of them may be 'required'")

    def test_the_candidates_are_reported_as_optional_not_as_required(self):
        names = {entry["name"] for entry in self.catalog_entries()}
        normalized = downloader.normalize_model_entries(
            llm_chat.load_models_config(), minimax=False, yue2=False, sheetsage2=False,
            flux2=False, flashsr=False, whisper=False, llm=True)
        checked = {entry["name"] for entry in normalized if entry.get("name") in names}
        self.assertEqual(checked, names, "every candidate stays visible in the check")
        with tempfile.TemporaryDirectory() as tmp:
            preflight = downloader.preflight_models(normalized, base_path=Path(tmp), auto_download=True)
        summary = preflight["summary"]
        self.assertEqual([name for name in summary["required_missing"] if name in names], [])
        self.assertEqual({name for name in summary["optional_missing"] if name in names}, names)
        self.assertTrue(summary["ok"], "a machine without a chat model is still a complete install")

    def test_the_model_check_never_starts_a_download_nobody_asked_for(self):
        names = {entry["name"] for entry in self.catalog_entries()}
        normalized = downloader.normalize_model_entries(
            llm_chat.load_models_config(), minimax=False, yue2=False, sheetsage2=False,
            flux2=False, flashsr=False, whisper=False, llm=True)
        entries = [entry for entry in normalized if entry.get("name") in names]
        self.assertTrue(entries)
        started = []

        def refuse(*args, **kwargs):  # pragma: no cover - only runs on a regression
            started.append(args)
            raise AssertionError("the preflight downloaded a model nobody selected")

        original = downloader.download_file
        downloader.download_file = refuse
        try:
            with tempfile.TemporaryDirectory() as tmp:
                report = downloader.check_file_entries(entries, base_path=Path(tmp), auto_download=True)
        finally:
            downloader.download_file = original
        self.assertEqual(started, [])
        for item in report:
            self.assertEqual(item["status"], "missing", item)
            self.assertTrue(item["message"], f"{item['name']} must explain why nothing was fetched")

    def test_a_selected_model_is_fetched_before_it_is_loaded(self):
        name = self.catalog_entries()[0]["name"]
        entry = llm_chat._configured_llm_entry(name)
        self.assertIsNotNone(entry, "a catalog model must be resolvable by name")
        self.assertFalse(entry.get("no_auto_download"),
                         "the model was chosen, so this is where it is fetched")
        self.assertTrue(downloader.resolve_entry_url(entry))

        calls = []
        searched = []

        def fake_find(model_name):
            searched.append(model_name)
            return None if len(searched) == 1 else Path("downloaded") / model_name

        def fake_check(entries, base_path=None, auto_download=False):
            calls.append({"entries": list(entries), "auto_download": auto_download})
            return [{"name": name, "target": "models/llm", "status": "downloaded", "message": ""}]

        original_find = llm_chat._find_model_path
        original_check = llm_chat.check_file_entries
        original_import = llm_chat._import_llama_cpp
        original_signature = llm_chat._model_signature
        llm_chat._find_model_path = fake_find
        llm_chat.check_file_entries = fake_check
        llm_chat._model_signature = lambda path: "test"
        llm_chat._import_llama_cpp = lambda: types.SimpleNamespace(
            __version__="test",
            Llama=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("llama.cpp was reached")))
        try:
            with self.assertRaises(RuntimeError) as caught:
                llm_chat._get_model(name, True, n_ctx=2048, chat_format="none")
        finally:
            llm_chat._find_model_path = original_find
            llm_chat.check_file_entries = original_check
            llm_chat._import_llama_cpp = original_import
            llm_chat._model_signature = original_signature
        self.assertEqual(len(calls), 1, "the missing model must be fetched exactly once")
        self.assertTrue(calls[0]["auto_download"], "the fetch is what 'auto_download' means here")
        self.assertEqual([e["name"] for e in calls[0]["entries"]], [name])
        self.assertEqual(searched, [name, name], "the path is resolved again after the download")
        self.assertIn("llama.cpp was reached", str(caught.exception),
                      "the download must finish before the model is loaded")

    def test_auto_download_off_says_so_instead_of_fetching(self):
        name = self.catalog_entries()[0]["name"]
        called = []

        def fake_check(*args, **kwargs):  # pragma: no cover - must not be reached
            called.append(True)
            return []

        original_find = llm_chat._find_model_path
        original_check = llm_chat.check_file_entries
        original_import = llm_chat._import_llama_cpp
        llm_chat._find_model_path = lambda model_name: None
        llm_chat.check_file_entries = fake_check
        llm_chat._import_llama_cpp = lambda: types.SimpleNamespace(__version__="test")
        try:
            with self.assertRaises(RuntimeError) as caught:
                llm_chat._get_model(name, False)
        finally:
            llm_chat._find_model_path = original_find
            llm_chat.check_file_entries = original_check
            llm_chat._import_llama_cpp = original_import
        self.assertIn("auto-download is disabled", str(caught.exception))
        self.assertEqual(called, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
