# Model advisor · what suits this machine

The toolkit's catalog can download many models, and "which one should I get?" has a
different answer on a 6 GiB laptop than on a 24 GiB workstation. This node answers it
for the machine it runs on:

- it reads the detected hardware (CPU cores, RAM, and the class of the accelerator),
- it walks the catalog (`models_config.json`) task by task — language model, song
  generation, audio-to-score, lyrics transcription, artwork, super-resolution,
- and it reports, per task, **which file fits**, how large it is, and how well it is
  **rated for that task** (1–5 stars).

The report is shown in the node itself (Markdown preview), written to the ComfyUI log,
and emitted as JSON on the second output for an app UI or a script.

## The two questions it answers

**Does it fit?** Each file gets a verdict computed from its size against the *free*
budget of the device it would be loaded onto:

- `fits` — weights plus a stated margin (`max(2 GiB, 20 %)`) fit the free memory,
- `tight` — the weights alone fit, but the margin for context, activations and staging
  buffers does not; expect offload or a shorter context,
- `too large` — not even the weights fit,
- `unchecked` — the free memory could not be read. Nothing is then claimed to fit;
  "unknown" and "nothing" are different answers.

Because a diffusion model and its text encoder are loaded by the *same* run, every
group is also checked as a **combination**. On a 12 GiB card the full-precision FLUX.2
pair is reported as too large, together with the smaller set (fp8 diffusion + fp4 text
encoder) that does fit.

**How good is it?** The stars judge suitability for *this toolkit's* task — holding a
long structured prompt, producing the `[Caption]`/`[Lyrics]`/`[Title]`/`[Image_Prompt]`
sections, singing-accurate transcription, generation quality. They are a documented
judgement, not a benchmark, and never a claim about the model in general. An entry
without a rating is shown unrated rather than given one.

## What it does not do

It reports. It does not download, does not change a widget, and does not select
anything for you: the model check (`MiniMaxModelAutodownload`) and the LLM node are the
places that transfer a file, and only when asked. Selection happens where it always
did — in the model dropdowns of the workflow nodes.

## Inputs

- **detail** — `summary` shows one recommended file per task plus the reason for the
  choice and the combination verdict; `full` lists every alternative with its size,
  stars, intended hardware class and measured verdict.
- **resources_json** (optional) — a resource snapshot as JSON (the format the
  diagnostics script writes). Connect it to advise for a *different* machine, for
  example when preparing a workflow for someone else's PC. Empty means "detect this
  machine".
