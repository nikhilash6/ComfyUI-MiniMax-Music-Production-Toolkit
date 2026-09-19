"""Explicit UI actions for LLM configuration.

Secrets live in server RAM, or - only when the user switches "Keep API key after
restart" on - in a file the server owns. Either way they stay out of the browser and
out of the workflow.
"""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from .llm_providers import LOCAL, CLOUD, connection, remember_key, forget_key, forget_saved_key, list_remote_models

_REGISTERED = False
PATH = "/minimax_music_toolkit/llm"


def _base_of(options):
    """The exact API base a request is about, or "" when it cannot be resolved.

    Used by the clear action, which must also reach a stored key - and an address that
    cannot be resolved (the integrated backend, an unknown provider) has none.
    """
    try:
        return connection(options["backend"], options["local_provider"],
                          options["cloud_provider"], options["server_url"])[1]
    except (ValueError, RuntimeError):
        return ""


def register_routes():
    global _REGISTERED
    if _REGISTERED:
        return
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get(PATH + "/providers")
    async def providers(request):
        return web.json_response({"local": LOCAL, "cloud": CLOUD})

    @PromptServer.instance.routes.post(PATH + "/configure")
    async def configure(request):
        # Configuration is a same-origin UI operation, never a cross-site form.
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            return web.json_response({"error": "Cross-origin configuration is not allowed."}, status=403)
        if request.content_type != "application/json":
            return web.json_response({"error": "Expected JSON."}, status=415)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("Expected a configuration object.")
            action = body.get("action")
            permanent = bool(body.get("permanent"))
            options = {name: str(body.get(name, "")) for name in (
                "backend", "local_provider", "cloud_provider", "server_url", "api_key_env", "credential_id")}
            if action == "clear_key":
                # Clear means clear: a stored key for this address goes with the session
                # one, otherwise a secret nobody can see would outlive the button meant
                # to remove it.
                forget_key(options["credential_id"])
                forget_saved_key(_base_of(options))
                return web.json_response({"ok": True})
            if action == "set_key":
                _, base, _, _ = connection(options["backend"], options["local_provider"], options["cloud_provider"], options["server_url"])
                handle = remember_key(base, body.get("key", ""), options["credential_id"], permanent)
                return web.json_response({"credential_id": handle})
            if action == "models":
                models = await asyncio.to_thread(list_remote_models, **options, permanent_key=permanent)
                return web.json_response({"models": models})
            raise ValueError("Unknown configuration action.")
        except (ValueError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    _REGISTERED = True
