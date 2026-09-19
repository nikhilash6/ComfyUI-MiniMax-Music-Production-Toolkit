"""One place to configure every LLM call in the toolkit.

A run makes up to three language-model calls - the song request, the Cover Studio
plan and the Cover Studio transformation - and each of them used to carry its own
copy of the same ~28 settings. Editing the provider, the model or a sampler value
meant editing it three times, and a workflow where two calls disagreed failed in
ways that were hard to see.

``MiniMaxLLMSettings`` holds those settings once and hands them out as
``llm_config_json``. Connect it to the ``llm_config_json`` input of any
:class:`~.llm_chat.MiniMaxLLMChat` node and its values win over that node's own
widgets, field by field: a setting the JSON does not carry still comes from the
receiving node, so an older graph keeps working.

The widget list is derived from the chat node rather than copied, so the two can
never offer different models, providers or formats. Per-call settings stay on the
chat node: ``enabled`` (skip this call), ``user_text``, ``system_prompt`` and
``reset_session`` (whether the call reuses the previous conversation).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from .toolkit_logging import get_logger

LOGGER = get_logger("llm_config")

CONFIG_SCHEMA = "music_llm_config_v1"

#: Per-call settings that belong to the chat node, never to the central node.
PER_CALL_FIELDS = ("enabled", "user_text", "system_prompt", "session_id", "reset_session")


def _chat_input_specs() -> Dict[str, Dict[str, Any]]:
    """The chat node's widget specs, minus the per-call ones, as ``{name: kwargs}``."""
    from .llm_chat import MiniMaxLLMChat

    data = MiniMaxLLMChat.INPUT_TYPES()
    specs: Dict[str, Dict[str, Any]] = {}
    for section in ("required", "optional"):
        for name, spec in (data.get(section) or {}).items():
            if name in PER_CALL_FIELDS:
                continue
            if len(spec) > 1 and isinstance(spec[1], dict) and spec[1].get("forceInput"):
                continue  # sockets (the prompt text) are per call as well
            specs[name] = spec
    return specs


def central_field_names() -> Tuple[str, ...]:
    """Field names the central node carries, in the chat node's own order."""
    return tuple(_chat_input_specs())


def build_config(values: Dict[str, Any]) -> Dict[str, Any]:
    """The payload handed to the chat nodes: only known fields, plus a schema tag."""
    config: Dict[str, Any] = {"schema": CONFIG_SCHEMA}
    for name in central_field_names():
        if name in values:
            config[name] = values[name]
    return config


def parse_config(payload: Any) -> Dict[str, Any]:
    """Read a config payload. Anything unreadable is *ignored*, never fatal.

    A socket can carry an empty string (nothing connected yet) or text from an
    older release. Neither may take a run down, so an unusable payload simply
    leaves the receiving node's own widgets in charge.
    """
    if not payload:
        return {}
    if isinstance(payload, dict):
        data = payload
    else:
        try:
            data = json.loads(str(payload))
        except (ValueError, TypeError):
            LOGGER.warning("Ignoring unreadable llm_config_json; using the local widgets instead.")
            return {}
    if not isinstance(data, dict):
        return {}
    known = set(central_field_names())
    return {name: data[name] for name in known if name in data}


def field_defaults() -> Dict[str, Any]:
    """The chat node's own defaults, so a resolved setting equals its widget value."""
    defaults: Dict[str, Any] = {}
    for name, spec in _chat_input_specs().items():
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        defaults[name] = options.get("default")
    return defaults


def resolve(payload: Any, values: Dict[str, Any]) -> Dict[str, Any]:
    """Every central field: the local widget value first, the config payload on top.

    Only the fields the payload actually carries are replaced, so a half-filled central
    node never blanks a setting on the receiving node - it just keeps its own value.
    Fields the caller does not pass at all fall back to the chat node's own default.
    """
    resolved = field_defaults()
    for name in resolved:
        if name in values and values[name] is not None:
            resolved[name] = values[name]
    overrides = parse_config(payload)
    resolved.update(overrides)
    return resolved


class MiniMaxLLMSettings:
    DESCRIPTION = (
        "Central LLM configuration for the whole workflow: provider, model, context and sampler "
        "settings for every LLM call, entered once. Connect the output to the llm_config_json "
        "input of the LLM chat nodes; its values then win over their own widgets, field by field. "
        "Per-call settings (enabled, the prompt text, session reset) stay on the chat nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        specs = _chat_input_specs()
        # Everything is `required` here: the node is the configuration, so a value it
        # does not carry is one the receiving nodes keep for themselves.
        return {"required": specs}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("llm_config_json",)
    FUNCTION = "configure"
    CATEGORY = "Music Production Toolkit/llm"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # The receiving node validates the effective values; this node only carries
        # them. Validating again here would refuse a config while the node it feeds
        # is not even connected.
        return True

    def configure(self, **values):
        config = build_config(values)
        LOGGER.info(
            "LLM settings: backend=%s, model=%s, context=%s, max_tokens=%s - sent to every connected LLM call.",
            config.get("backend"), config.get("model"), config.get("n_ctx"), config.get("max_tokens"),
        )
        return (json.dumps(config, ensure_ascii=False),)


NODE_CLASS_MAPPINGS = {"MiniMaxLLMSettings": MiniMaxLLMSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxLLMSettings": "LLM settings · central (one place for every call)"}
