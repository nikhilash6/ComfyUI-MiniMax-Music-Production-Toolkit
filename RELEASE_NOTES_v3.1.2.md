# Music Production Toolkit 3.1.2

A Swiss Army knife for local music production on ComfyUI.

A patch on top of 3.1.1: the toolkit now looks at the PC it runs on and says which
models suit it, the language-model settings are entered once instead of three times,
long stages report their progress the way ComfyUI's own nodes do, and two broken
downloads were fixed.

## Setup now picks models for your PC

The catalog always knew *how* to download a model; nothing knew whether that model
suits the machine in front of you.

- **New node `Model advisor · what suits this machine`** (in the shipped workflow next
  to the model check). It reads the detected hardware — CPU cores, RAM, the class of
  the accelerator — and reports, per task, which file fits, how large it is and how it
  is rated. The verdict is measured against the **free** budget with a stated margin,
  so "the weights fit" and "the weights plus context fit" stay two different answers,
  and an unreadable device is reported as unknown instead of being guessed.
- **Every task is checked as a combination.** A diffusion model and its text encoder
  load in the same run, so on a 12 GiB card the full-precision FLUX.2 pair is reported
  as too large *together with* the fp8/fp4 set (7.7 GiB) that does fit.
- **Star ratings for the job at hand.** Every catalog entry carries a 1–5 star rating
  with its reason — suitability for *this* toolkit's tasks (structured caption, lyrics
  and score output, generation quality, transcription), not a benchmark of the model in
  general. The stars are also printed in the model check's report and documented in
  [installation](INSTALLATION.md).
- **Smaller alternatives that actually exist**, each verified against its repository
  (pinned commit, exact byte size) and loadable with ComfyUI's own loaders — no extra
  custom node:
  - language models: 2B, 4B, 9B-distill, 12B and a plain `llama`-architecture 8B for
    older `llama.cpp` builds, on top of the existing 9B/12B/27B candidates;
  - MiniMax Music 3: the int8 diffusion model (2.3 GiB) and the pruned int8 text
    encoder — the pair that makes a 12–16 GiB card workable;
  - YuE2: the 3B int8 checkpoint (3.7 GiB);
  - FLUX.2 klein: fp8 diffusion plus the fp4 text encoder;
  - Whisper: `large-v3-turbo` (1.5 GiB) and a turbo int8 (0.8 GiB) beside the
    reference large-v3.
- **A model you selected downloads itself.** The Whisper node now fetches the
  checkpoint folder you picked (the group checkbox keeps fetching only the default), and
  the LLM node fetches the GGUF you selected — the download happens where the decision
  was made, not by a checkbox that would pull in several alternatives.
- **What to expect from your PC** is now a table in the README: per configuration
  (CPU only, 6–8, 10–12, 16, 24 and 32+ GiB) it names the fitting music model, language
  model, Whisper checkpoint and artwork pair, the speed band per stage and the rating of
  each choice — marked as a starting point, not a measurement.

## Enter the language-model settings once

- **`LLM settings · central`** holds provider, model, context and sampler values once
  and hands them to every LLM chat node in the workflow; its values win field by field,
  so a half-filled central node changes nothing it was not asked to change, and an
  unreadable payload is ignored with a log line instead of stopping a run. Per-call
  settings (`enabled`, the prompt texts, `reset_session`) stay on the chat node.
- **A leftover model name no longer blocks a run.** ComfyUI validates every node's
  widgets before execution and hands a linked input over as a placeholder (never as the
  payload), so a connected settings node is treated as the decision — the model is
  checked when it is loaded.

## Runs are easier to follow

- **The LLM and FlashSR stages draw the bar ComfyUI's own nodes draw** (tqdm, the one
  YuE2 uses): one line that updates in place with count, percentage, elapsed time,
  remaining time and rate — `LLM streaming:   8%|# | 1958/24576 [01:15<14:24, 26.1token/s]`.
  No more one log line per step; the log keeps the start line and one summary per stage.
- **A refinement stage that cannot run switches itself off.** If FlashSR's weights are
  neither installed nor downloadable, the stage logs why and passes the audio through
  unchanged instead of ending a run that can still produce the song.
- **A damaged source file is refused before the run starts, with its name in the
  message.** The cover transcription node decodes the file once and reports
  `the source audio 'x.mp3' cannot be decoded past 2:42 (…)` instead of letting
  ComfyUI's loader fail 1.5 seconds in with a traceback that never named the file.

## Fixed in 3.1.2

- **The FlashSR weights pointed at a repository that stopped answering.** Every request
  to `jakeoneijk/FlashSR_weights` returns HTTP 401 repo-wide, so a fresh install could
  not download the three weights at all. The catalog now reads the identical files —
  same byte sizes — from the authors' repository.
- **A chat model could be listed and still refuse to download itself.** The loader asked
  the catalog entry for a `url` field, which the current entries do not have (they name
  a repository, a pinned commit and a remote filename), so every catalog model answered
  "no download URL is configured".
- **Whisper alternatives were unreachable** although they were in the catalog: the group
  checkbox fetches only the default checkpoint by design, and nothing fetched the selected
  one. The lyrics node does now, and a checkpoint folder that exists without `model.bin`
  is reported as an incomplete download instead of being handed to the engine.
- **Layout:** the Model advisor sits inside `01 · START / Files & models`, and one node
  that had been overlapping its neighbour was nudged back into place.

## Assets

- `Music_Production_Toolkit_v3.1.2.json`
- `Music_Production_AudioEnhance_v3.1.2.json`
- `ComfyUI-MiniMax-Music-Production-Toolkit-v3.1.2.zip`
- `SHA256SUMS.txt`

The previous release archives stay where they are.

## Upgrading

Replace the toolkit folder, restart ComfyUI and refresh the browser. Re-import the two
bundled workflows to get the new `Model advisor` node and the central LLM settings node;
everything else loads unchanged — no node input was renamed, reordered or removed, and
an existing workflow keeps working without the new nodes (the settings node is optional
everywhere it is wired).

Model files you already have stay valid; the new entries are *catalog* entries, so
nothing is downloaded until you select or check it.
