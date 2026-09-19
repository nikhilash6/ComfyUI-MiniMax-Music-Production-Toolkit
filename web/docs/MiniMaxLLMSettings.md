# LLM settings · central (one place for every call)

A run can make up to three LLM calls — the song request, the Cover Studio plan and the
Cover Studio transformation. They used to carry an identical copy of the same ~28
settings each, so changing the provider, the model or a sampler value meant editing it
three times, and a workflow whose copies disagreed failed in ways that were hard to see.

This node holds those settings **once**. Its single output goes to the `llm_config_json`
input of every LLM chat node in the workflow.

## Which settings live here

Everything that describes *where the text comes from and how it is generated*:

- **Backend and connection** — `backend`, `local_provider`, `cloud_provider`,
  `server_url`, `remote_model`, `api_key_env`, `credential_id`, `remote_max_tokens`,
  `request_timeout`, `permanent_key`. The provider buttons (**Set API key**, **Find models / test
  connection**, **Connection setup / status**) work here exactly as they do on the chat
  node, and the fields that do not apply to the selected backend are hidden the same way.
  `permanent_key` is the **Keep API key after restart** switch: off, a key entered with
  **Set API key** lives in ComfyUI's memory for that session; on, it is stored on this
  computer, bound to the exact API address, and reused after a restart — so the key is
  entered once instead of every session. It never reaches the workflow either way. The
  settings node is usually the better place for it: it feeds every connected call, so
  one switch covers the song request and both Cover Studio calls.
- **Model** — the GGUF for the integrated backend, or the server model ID for external
  ones.
- **Budgets** — `max_tokens`, `n_ctx`.
- **Sampling** — `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`,
  `presence_penalty`, `frequency_penalty`, `seed`, `thinking`, `chat_format`.
- **GGUF runtime** — `n_gpu_layers`, `split_mode`, `tensor_split`, `main_gpu`,
  `tensor_parallel`, and `auto_download`.

## What stays on the chat node

Three settings are about a single call, not about the model:

- `enabled` — skip this call while the others still run.
- `user_text` and `system_prompt` — the text this call receives.
- `reset_session` — whether the call continues the previous conversation.
- `llm_config_json` — the input this node feeds.

## How the two nodes combine

Connect the output and the central values win **field by field**: a setting the config
carries replaces the chat node's own, and a setting it does not carry keeps the value
configured on the receiving node. That is why an older graph, or a chat node used on its
own, behaves exactly as before — the connection is optional everywhere.

Only the fields the central node actually holds are sent, so a partially configured
central node never blanks a setting somewhere else. A config that cannot be read (an
empty string, or text from an older release) is ignored with a log line instead of
stopping the run.

ComfyUI checks every node's own widgets before it executes anything, and a linked
input does not exist yet at that point. The chat node therefore treats a connection as
the decision: a model name left in its own dropdown — a file that was deleted, or a
workflow from another machine — no longer refuses the run, and the model that is used
is the one the settings node sends. Disconnect the node and the chat node validates its
own value again.

## When to connect it

- **Several calls, one provider.** One place for the provider, the key and the model.
- **Comparing models.** Change the model once instead of three times.
- **Cover runs.** The plan and the transformation calls should use the same model as the
  song request; connecting this node makes that the default instead of a manual habit.
- **Per-call differences are still possible.** Disconnect the node, or leave a field out
  of it, and the chat nodes keep their own values.

The startup line reports which source was used for each call, so a misconfigured graph
says so instead of quietly generating text with a different model.
