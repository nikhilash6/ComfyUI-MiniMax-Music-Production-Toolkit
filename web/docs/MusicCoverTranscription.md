# Cover song · SheetSage2 transcription

Expands native LoadAudio, AudioEncoderLoader and SheetSage2AudioToABC only when
**YuE2 Cover** is selected. The model-check report input ensures configured
downloads finish before the encoder is loaded. Other models return an empty
string without loading source audio or SheetSage2.

Connect `cover_abc` to *Cover song · instrumental score / phrase map*. That node adapts the
score to the selected cover lyrics mode - an instrumental cover rewrites the
vocal melody line into an instrument part - and the LLM and the native generator
both receive that adapted score. For the two lyric-writing modes the score is
handed through unchanged, and no new ABC is generated in this mode.

SheetSage2 extracts music, not the original sung words: the score carries two
melodic voices (`Vocal` and `Ins`), structure labels and chord symbols, but no
lyric text. Use the *original lyrics* cover mode (Whisper) to transcribe the
original words, or supply words in the brief. An empty transcription stops the
cover run with an error.

Before the graph is built, the selected file is decoded once end to end. A file whose
decoder gives up - a truncated download, or a broken frame in the middle - is refused
with a message naming the file and how far it got, instead of failing later inside
ComfyUI's `LoadAudio` with a traceback that names nothing. A normal song costs about
0.2 s to check. See [troubleshooting](../../TROUBLESHOOTING.md#a-cover-run-dies-with-invaliddataerror-or-header-missing).

## Inputs

- **model_profile_json** - decides whether anything is transcribed at all: only a YuE2 Cover
  profile loads the source audio.
- **cover_source_json** - the selected file and the shared transcription/generation mode.
- **model_check_report** - SheetSage2 preflight result, so a missing model is reported before
  the load is attempted.

Other song models load nothing, which keeps a new-song run independent of the source audio.
