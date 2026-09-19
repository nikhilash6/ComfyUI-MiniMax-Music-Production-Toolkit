"""Which model suits this machine (R03): hardware class, candidates, star ratings.

The toolkit ships one catalog of downloadable artifacts (``models_config.json``).
Every entry names its repository, a pinned revision and its byte size - but a
catalog says nothing about whether a *particular machine* should download a
particular file. This module answers that, per detected hardware:

* **What kind of machine is this?** :func:`hardware_summary` reports CPU cores,
  RAM and the accelerator class (``cpu``, ``vram_8``, ...) from
  :mod:`resource_profiles` - an unreadable device stays ``vram_unknown`` instead
  of being guessed.
* **What fits?** Each catalog entry gets a measured verdict - ``ok``, ``tight``,
  ``too_large`` or ``unknown`` - from its file size against the *free* budget of
  the device it would be loaded onto. Several GPUs are never added up: they are
  separate memory pools.
* **How good is it for the job?** The catalog carries a 1-5 star rating per file
  (``rating``) with a one-line reason (``rating_note``) - a judgement about
  suitability for this toolkit's tasks, not a benchmark and not a quality claim
  about the model in general. The stars are shown as-is; they never override a
  choice.

Honesty rules kept here, matching the rest of the toolkit:

* a file size is a **lower bound** for the resident footprint (context/KV,
  activations, staging and backend overhead come on top), so a verdict is a
  statement about a *check*, never a promise - and the margin is stated;
* a verdict of "too large" is not a refusal: the user may still download it;
* nothing here downloads, writes or changes a setting. The advisor reports; the
  model check (``MiniMaxModelAutodownload``) and the LLM node are the places that
  transfer a file, and only when asked;
* a missing classification is reported as such. "Unknown" and "nothing" are
  different answers.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import resource_profiles
from .model_downloader import (
    check_file_entries,
    load_models_config,
    normalize_model_entries,
    resolve_target,
    stars,
)
from .toolkit_logging import get_logger

LOGGER = get_logger("model_advisor")

GIB = 1024 ** 3

#: Extra room a loaded model needs on top of its weights: activations, the
#: KV/context state, staging buffers and fragmentation. Applied as
#: ``max(2 GiB, 20 %)`` and always *stated* in the verdict.
OVERHEAD_MIN_BYTES = 2 * GIB
OVERHEAD_FRACTION = 0.20

# Catalog group -> (label, ((role, role label), ...)). The order is the order a
# run needs the groups in, so the report reads like the workflow.
CATALOG_GROUPS: Tuple[Tuple[str, str, Tuple[Tuple[str, str], ...]], ...] = (
    ("llm", "Caption, lyrics and prompt · language model",
     (("chat", "chat model"),)),
    ("minimax", "Song generation · MiniMax Music 3",
     (("diffusion", "diffusion model"), ("text_encoder", "text encoder"), ("vae", "VAE"))),
    ("yue2", "Song generation · YuE2 (also the cover engine)",
     (("checkpoint", "checkpoint"),)),
    ("sheetsage2", "Audio → score · SheetSage2 (YuE2 covers)",
     (("audio_encoder", "audio encoder"),)),
    ("whisper", "Lyrics from the source audio · Whisper",
     (("asr", "speech model"),)),
    ("flux2", "Cover artwork · FLUX.2",
     (("diffusion", "diffusion model"), ("text_encoder", "text encoder"), ("vae", "VAE"))),
    ("flashsr", "Audio super-resolution · FlashSR",
     (("weights", "weights"),)),
)

#: Groups whose files do not carry a ``role`` in the catalog.
_GROUP_DEFAULT_ROLE = {
    "yue2": "checkpoint",
    "sheetsage2": "audio_encoder",
    "whisper": "asr",
    "flashsr": "weights",
    "llm": "chat",
}

#: The first role of a group is the one whose size decides whether that group is
#: usable at all (a small diffusion model is useless without its text encoder).
_PRIMARY_ROLE = {"flashsr": "weights"}


#: Groups whose entries form one artifact per target folder: a Whisper checkpoint is
#: five files in one folder, FlashSR is three weights that only work together, and every
#: file of a group lives in that same folder. Every other group lists one artifact per
#: file, because its files sit in the same folder as *alternatives* (three MiniMax
#: diffusion quantizations are three models, not one).
PACKED_BY_FOLDER = ("whisper", "flashsr")


def stars_row(rating: Optional[int]) -> str:
    """Star row for a rating - the same rendering the model check uses."""
    return stars(rating)


def _group_of_each_file(config: Dict[str, Any]) -> Dict[str, str]:
    """``{file name: group}`` for every file the catalog lists."""
    mapping: Dict[str, str] = {}
    for group, _label, _roles in CATALOG_GROUPS:
        body = config.get(group) or {}
        if not isinstance(body, dict):
            continue
        files = list(body.get("files") or [])
        weights = body.get("weights")
        if isinstance(weights, dict):
            files += list(weights.get("files") or [])
        for entry in files:
            if isinstance(entry, dict) and entry.get("name"):
                mapping[str(entry["name"])] = group
    return mapping


def catalog_candidates(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every catalog *artifact* as a flat, group-aware candidate (no disk access).

    An artifact is what a run loads as one thing: a single file (a GGUF, a diffusion
    model, a VAE) or a pack of files that only work together - a Whisper checkpoint is
    five files in one folder, and offering them separately would suggest that
    ``preprocessor_config.json`` is a speech model. Files are therefore packed by
    ``(group, target folder)`` and share one rating and one size.

    Optional artifacts are included on purpose: the advisor's job is to show what
    *could* be downloaded, including the smaller alternatives that the automatic
    checks deliberately leave alone.
    """
    config = config if isinstance(config, dict) else load_models_config()
    group_of = _group_of_each_file(config)
    labels = {group: label for group, label, _roles in CATALOG_GROUPS}
    defaults = dict(_GROUP_DEFAULT_ROLE)
    entries = normalize_model_entries(
        config,
        minimax=True, yue2=True, sheetsage2=True, flux2=True, flashsr=True, llm=True, whisper=True,
        include_optional=True,
    )

    packed: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
    for entry in entries:
        name = str(entry.get("name") or "")
        group = group_of.get(name)
        if not group:
            continue
        role = str(entry.get("role") or defaults.get(group, "file"))
        target = str(entry.get("target") or "")
        size = entry.get("bytes")
        try:
            size_bytes = int(size) if size else None
        except (TypeError, ValueError):
            size_bytes = None
        rating = entry.get("rating")
        try:
            rating_value = int(rating) if rating is not None else None
        except (TypeError, ValueError):
            rating_value = None
        key = (group, role, target) if group in PACKED_BY_FOLDER else (group, role, target, name)
        artifact = packed.get(key)
        if artifact is None:
            artifact = {
                "group": group,
                "group_label": labels.get(group, group),
                "role": role,
                "name": name,
                "files": [],
                "bytes": 0,
                "size_known": True,
                "rating": rating_value,
                "rating_note": str(entry.get("rating_note") or entry.get("note") or ""),
                "suits": list(entry.get("suits") or []),
                "optional": bool(entry.get("optional")),
                "no_auto_download": bool(entry.get("no_auto_download")),
                "target": target,
                "repo_id": entry.get("repo_id") or "",
                "revision": entry.get("revision") or "",
            }
            packed[key] = artifact
        artifact["files"].append(name)
        artifact["bytes"] += size_bytes or 0
        if size_bytes is None:
            artifact["size_known"] = False
        # A pack is described by its first (largest, model-defining) file; the packed
        # folder name is what the user sees in the model dropdown.
        if size_bytes and len(artifact["files"]) == 1:
            artifact["name"] = name

    candidates: List[Dict[str, Any]] = []
    for artifact in packed.values():
        if len(artifact["files"]) > 1 and artifact["target"]:
            artifact["name"] = Path(artifact["target"]).name
        if not artifact["size_known"]:
            artifact["bytes"] = None
        artifact["gib"] = round(artifact["bytes"] / GIB, 2) if artifact["bytes"] else None
        artifact["stars"] = stars(artifact["rating"])
        candidates.append(artifact)
    return candidates


def hardware_summary(resources: Optional[resource_profiles.ResourceSnapshot] = None) -> Dict[str, Any]:
    """The machine class plus the numbers a verdict is based on."""
    resources = resources if resources is not None else resource_profiles.detect_resources()
    hardware_class = resource_profiles.hardware_class(resources)
    accelerators = list(resources.accelerators)
    device = max(accelerators, key=lambda item: item.vram_total_bytes or 0) if accelerators else None
    budget: Optional[int] = None
    if device is not None and device.kind == "cuda":
        budget = resource_profiles.device_budget(resources, device.id)
    elif device is not None:  # mps uses unified memory
        budget = resources.ram_available_bytes
    if device is None:
        budget = resources.ram_available_bytes
    return {
        "class": hardware_class,
        "label": {
            "cpu": "CPU only",
            "vram_unknown": "accelerator with unreadable memory",
        }.get(hardware_class, f"≤ {hardware_class.split('_')[-1]} GiB VRAM class"),
        "device": device.id if device else "cpu",
        "device_name": device.name if device else "CPU",
        "vram_total_bytes": device.vram_total_bytes if device else None,
        "vram_free_bytes": device.vram_free_bytes if device else None,
        "ram_total_bytes": resources.ram_total_bytes,
        "ram_available_bytes": resources.ram_available_bytes,
        "cpu_count": resources.cpu_count,
        "gpu_count": len(accelerators),
        "budget_bytes": budget,
        "notes": list(resources.notes),
    }


def _needed_bytes(size_bytes: Optional[int]) -> Optional[int]:
    if not size_bytes:
        return None
    return int(size_bytes) + max(OVERHEAD_MIN_BYTES, int(int(size_bytes) * OVERHEAD_FRACTION))


def fit_verdict(candidate: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Whether one candidate fits the machine, with the numbers behind it.

    ``ok``       - the weights plus the stated overhead fit the free budget.
    ``tight``    - the weights alone fit; the overhead does not fit safely any
                   more (a smaller context, a freed GPU or an offload is needed).
    ``too_large``- not even the weights fit the free budget.
    ``unknown``  - the budget could not be read; nothing is claimed.
    """
    size = candidate.get("bytes")
    if size is None:
        return {"verdict": "unknown", "reason": "the catalog does not state a size for this file"}
    budget = summary.get("budget_bytes")
    if budget is None:
        return {
            "verdict": "unknown",
            "reason": "the free memory could not be read, so no fit is claimed - it is not a size test",
        }
    needed = _needed_bytes(size) or 0
    if needed <= budget:
        return {
            "verdict": "ok",
            "needed_bytes": needed,
            "free_after_bytes": int(budget) - needed,
            "reason": (
                f"weights {size / GIB:.2f} GiB + margin {max(OVERHEAD_MIN_BYTES, int(size * OVERHEAD_FRACTION)) / GIB:.2f} GiB "
                f"fit the free {budget / GIB:.2f} GiB"
            ),
        }
    if size <= budget:
        return {
            "verdict": "tight",
            "needed_bytes": needed,
            "missing_bytes": needed - int(budget),
            "reason": (
                f"the weights ({size / GIB:.2f} GiB) fit into {budget / GIB:.2f} GiB, but not with the "
                f"{max(OVERHEAD_MIN_BYTES, int(size * OVERHEAD_FRACTION)) / GIB:.2f} GiB margin for context and buffers"
            ),
        }
    return {
        "verdict": "too_large",
        "needed_bytes": needed,
        "missing_bytes": int(size) - int(budget),
        "reason": f"the weights alone ({size / GIB:.2f} GiB) exceed the free {budget / GIB:.2f} GiB",
    }


def combination_verdict(picks: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Whether one file per role fits *together* - the question a per-file check misses.

    A diffusion model and a text encoder are loaded by the same run: each may fit on
    its own while the pair does not. The verdict therefore sums the recommended files
    of a group and applies the same margin.
    """
    chosen = [pick for pick in picks if pick]
    if not chosen:
        return {"verdict": "unknown", "reason": "no candidate was chosen for this group"}
    sizes = [pick.get("bytes") for pick in chosen]
    if any(size is None for size in sizes):
        return {
            "verdict": "unknown",
            "names": [pick["name"] for pick in chosen],
            "reason": "at least one chosen file has no size in the catalog",
        }
    total = int(sum(sizes))
    budget = summary.get("budget_bytes")
    needed = _needed_bytes(total) or total
    names = [pick["name"] for pick in chosen]
    if budget is None:
        return {
            "verdict": "unknown",
            "names": names,
            "bytes": total,
            "reason": "the free memory could not be read, so no fit is claimed for the combination",
        }
    base = {
        "names": names,
        "bytes": total,
        "needed_bytes": needed,
        "budget_bytes": int(budget),
    }
    if needed <= budget:
        return {**base, "verdict": "ok",
                "reason": f"together {total / GIB:.2f} GiB + margin fit the free {budget / GIB:.2f} GiB"}
    if total <= budget:
        return {**base, "verdict": "tight",
                "reason": (f"together {total / GIB:.2f} GiB leave no room for the margin in "
                           f"{budget / GIB:.2f} GiB - expect offload or a shorter context")}
    return {**base, "verdict": "too_large",
            "reason": f"together {total / GIB:.2f} GiB exceed the free {budget / GIB:.2f} GiB"}


def file_status(config: Optional[Dict[str, Any]] = None,
                base_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Per-file status of the whole catalog, keyed by its absolute destination path.

    Stat and size check only - never a transfer. Keyed by path (not by file name),
    because ``model.bin`` exists in every Whisper folder.
    """
    entries = catalog_entries_for_check(config)
    report = check_file_entries(entries, base_path=base_path, auto_download=False)
    return {str(item.get("target") or ""): item for item in report}


def catalog_entries_for_check(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Normalized entries with their targets - the input ``check_file_entries`` needs."""
    config = config if isinstance(config, dict) else load_models_config()
    return normalize_model_entries(
        config,
        minimax=True, yue2=True, sheetsage2=True, flux2=True, flashsr=True, llm=True, whisper=True,
        include_optional=True,
    )


_VERDICT_RANK = {"ok": 0, "tight": 1, "unknown": 2, "too_large": 3}
_STATUS_RANK = {"present": 0, "missing": 1, "failed": 2}


def recommend_models(
    resources: Optional[resource_profiles.ResourceSnapshot] = None,
    config: Optional[Dict[str, Any]] = None,
    base_path: Optional[Path] = None,
    check_installed: bool = True,
) -> Dict[str, Any]:
    """The full advice: machine class, per-role candidates with stars, and picks.

    Every candidate carries its measured verdict and, when ``check_installed``
    is set, whether it is already on disk. The pick per role prefers something
    that is installed, then the best-rated candidate that fits - and says so when
    nothing fits instead of recommending the least-bad option silently.
    """
    summary = hardware_summary(resources)
    candidates = catalog_candidates(config)
    if check_installed:
        status = file_status(config, base_path=base_path)
        for candidate in candidates:
            folder = resolve_target(candidate["target"], base_path) if candidate["target"] else None
            present = 0
            messages: List[str] = []
            for name in candidate["files"]:
                info = status.get(str(folder / name) if folder else "") or {}
                if info.get("status") == "present":
                    present += 1
                elif info.get("message"):
                    messages.append(str(info["message"]))
            total = len(candidate["files"])
            candidate["installed"] = bool(total) and present == total
            candidate["files_present"] = present
            candidate["files_total"] = total
            candidate["status"] = "present" if candidate["installed"] else ("partial" if present else "missing")
            candidate["status_message"] = "" if candidate["installed"] else (
                f"{present} of {total} files present" if present else (messages[0] if messages else "")
            )
            candidate["installed_path"] = str(folder) if folder else ""
    for candidate in candidates:
        candidate["fit"] = fit_verdict(candidate, summary)

    groups: List[Dict[str, Any]] = []
    for group, label, roles in CATALOG_GROUPS:
        group_candidates = [c for c in candidates if c["group"] == group]
        if not group_candidates:
            continue
        role_entries: List[Dict[str, Any]] = []
        for role, role_label in roles:
            in_role = [c for c in group_candidates if c["role"] == role]
            if not in_role:
                continue
            in_role.sort(key=lambda c: (
                _VERDICT_RANK.get(c["fit"]["verdict"], 9),
                0 if c.get("installed") else 1,
                -int(c.get("rating") or 0),
                c.get("bytes") or 0,
            ))
            fitting = [c for c in in_role if c["fit"]["verdict"] in ("ok", "tight")]
            installed = [c for c in in_role if c.get("installed")]
            pick = None
            note = ""
            if installed:
                # Something is already there: never point away from it silently.
                pick = max(installed, key=lambda c: (int(c.get("rating") or 0),))
                best = next((c for c in fitting if int(c.get("rating") or 0) > int(pick.get("rating") or 0)), None)
                if best is not None:
                    note = (
                        f"{pick['name']} is installed; {best['name']} is rated higher "
                        f"({stars(best.get('rating'))}) and would fit as well."
                    )
            elif fitting:
                pick = max(fitting, key=lambda c: (int(c.get("rating") or 0), -int(c.get("bytes") or 0)))
            elif in_role:
                smallest = min(in_role, key=lambda c: c.get("bytes") or 1 << 62)
                note = (
                    "no candidate in this role fits the detected budget - the catalog has no smaller "
                    f"alternative yet ({smallest['name']} needs {smallest.get('gib')} GiB of weights)"
                )
            role_entries.append(
                {
                    "role": role,
                    "role_label": role_label,
                    "pick": pick["name"] if pick else "",
                    "pick_installed": bool(pick.get("installed")) if pick else False,
                    "note": note,
                    "candidates": [
                        {
                            "name": c["name"],
                            "files": c["files"],
                            "bytes": c["bytes"],
                            "files_present": c.get("files_present"),
                            "files_total": c.get("files_total"),
                            "gib": c["gib"],
                            "rating": c["rating"],
                            "stars": c["stars"],
                            "rating_note": c["rating_note"],
                            "fit": c["fit"],
                            "installed": bool(c.get("installed")),
                            "status": c.get("status"),
                            "status_message": c.get("status_message"),
                            "suits": c["suits"],
                            "optional": c["optional"],
                            "repo_id": c["repo_id"],
                            "files_list": c["files"],
                        }
                        for c in in_role
                    ],
                }
            )
        picks = [next((c for c in role["candidates"] if c["name"] == role["pick"]), None)
                 for role in role_entries]
        combination = combination_verdict(picks, summary)
        if combination.get("verdict") in ("tight", "too_large"):
            # Name the smallest complete set that does fit, instead of only complaining:
            # "does not fit" is only useful with the alternative next to it.
            smallest = []
            for role in role_entries:
                options = [c for c in role["candidates"] if c.get("bytes")]
                if options:
                    smallest.append(min(options, key=lambda c: c["bytes"]))
            if len(smallest) == len(role_entries):
                alternative = combination_verdict(smallest, summary)
                if (alternative.get("verdict") in ("ok", "tight")
                        and sorted(alternative.get("names") or []) != sorted(combination.get("names") or [])):
                    combination["suggestion"] = {
                        "names": alternative["names"],
                        "bytes": alternative["bytes"],
                        "verdict": alternative["verdict"],
                        "reason": alternative["reason"],
                    }
        groups.append({"group": group, "label": label, "roles": role_entries,
                       "machine_note": (
                           "No accelerator: generating music or artwork on the CPU is extremely slow - the "
                           "text and audio paths stay usable, and the model check can still fetch these files."
                           if summary["class"] == "cpu" and group in ("minimax", "yue2", "flux2") else ""
                       ),
                       "combination": combination})

    caveats = [
        "A file size is a lower bound: context, activations, staging and backend overhead come on top - the "
        "margin in each verdict is stated, not hidden.",
        "The star rating judges suitability for this toolkit's tasks (structured caption/lyrics output, "
        "generation quality, transcription) and is not a benchmark of the model in general.",
        "Nothing here downloads or changes a setting. The model check and the LLM node fetch a file only when asked.",
    ]
    if summary["class"] == "vram_unknown":
        caveats.append("The device memory could not be read, so no candidate is claimed to fit.")
    if summary["gpu_count"] > 1:
        caveats.append(f"{summary['gpu_count']} accelerators are separate memory pools; budgets are per device.")
    if summary["class"] == "cpu":
        caveats.append("No accelerator: generation runs on the CPU and is slow; the audio and text paths stay usable.")
    return {
        "schema": "music_model_advice_v1",
        "machine": summary,
        "groups": groups,
        "caveats": caveats,
        "counts": {
            "candidates": len(candidates),
            "fits": sum(1 for c in candidates if c["fit"]["verdict"] == "ok"),
            "installed": sum(1 for c in candidates if c.get("installed")),
        },
    }


def _candidate_line(candidate: Dict[str, Any]) -> str:
    marker = {"ok": "fits", "tight": "tight", "too_large": "too large", "unknown": "unchecked"}[
        candidate["fit"]["verdict"]
    ]
    installed = " · installed" if candidate.get("installed") else (
        f" · {candidate['files_present']}/{candidate['files_total']} files present"
        if candidate.get("files_present") else ""
    )
    size = f"{candidate['gib']} GiB" if candidate.get("gib") is not None else "size unknown"
    files = f", {len(candidate['files'])} files" if len(candidate.get("files") or []) > 1 else ""
    return f"{candidate['stars']} {candidate['name']} ({size}{files}, {marker}{installed})"


def format_advisor_lines(report: Dict[str, Any], detail: str = "summary") -> List[str]:
    """The readable report: machine first, then one block per group."""
    machine = report.get("machine") or {}
    if machine.get("class") == "cpu":
        where = f"CPU only ({machine.get('cpu_count') or '?'} cores)"
    else:
        total = machine.get("vram_total_bytes")
        free = machine.get("vram_free_bytes")
        total_text = f"{total / GIB:.1f} GiB" if total else "unknown"
        free_text = f"{free / GIB:.1f} GiB free" if free else "free memory unknown"
        where = f"{machine.get('device_name')} ({total_text}, {free_text})"
        if machine.get("gpu_count", 0) > 1:
            where += f" + {machine['gpu_count'] - 1} more"
    ram = machine.get("ram_total_bytes")
    ram_text = f"{ram / GIB:.1f} GiB RAM" if ram else "RAM unknown"
    counts = report.get("counts") or {}
    lines = [
        f"Model advisor - {where}, {ram_text} [class {machine.get('class')}]",
        f"  catalog: {counts.get('candidates', 0)} artifacts, {counts.get('fits', 0)} fit this machine, "
        f"{counts.get('installed', 0)} installed",
    ]
    for group in report.get("groups") or []:
        lines.append(f"  {group['label']}")
        for role in group.get("roles") or []:
            pick = role.get("pick")
            if pick:
                winner = next(c for c in role["candidates"] if c["name"] == pick)
                lines.append(f"    {role['role_label']}: {_candidate_line(winner)}")
            else:
                lines.append(f"    {role['role_label']}: no fitting candidate")
            if role.get("note"):
                lines.append(f"      note: {role['note']}")
        combination = group.get("combination") or {}
        if combination.get("names"):
            total = combination.get("bytes")
            size = f"{total / 1024 ** 3:.2f} GiB" if total else "size unknown"
            lines.append(f"    together ({size}): {combination.get('verdict')} - {combination.get('reason')}")
            suggestion = combination.get("suggestion") or {}
            if suggestion.get("names"):
                alt = suggestion.get("bytes")
                alt_size = f"{alt / 1024 ** 3:.2f} GiB" if alt else "size unknown"
                lines.append(
                    f"      smaller set that fits ({alt_size}): {', '.join(suggestion['names'])}"
                )
        if group.get("machine_note"):
            lines.append(f"    note: {group['machine_note']}")
            if detail == "full":
                for candidate in role["candidates"]:
                    if candidate["name"] == pick:
                        continue
                    lines.append(f"      alt: {_candidate_line(candidate)}")
                    if candidate.get("suits"):
                        lines.append(f"           intended for: {', '.join(candidate['suits'])}")
                    if candidate.get("rating_note"):
                        lines.append(f"           why: {candidate['rating_note']}")
            elif role["candidates"]:
                first = role["candidates"][0]
                if first["name"] != pick and first.get("rating_note"):
                    lines.append(f"      {first['stars']} {first['name']}: {first['rating_note']}")
    for caveat in report.get("caveats") or []:
        lines.append(f"  note: {caveat}")
    return lines


class MiniMaxModelAdvisor:
    """Report which models suit this machine, with a rating per candidate."""

    DESCRIPTION = (
        "Reads the detected hardware (CPU, RAM, accelerator class) and reports, per task, which catalog "
        "models fit it, how large they are and how well they are rated for that task (1-5 stars). "
        "It reports only: nothing is downloaded or changed. Use the model check or the LLM node to fetch a "
        "file, and the model dropdowns in the workflow to select it. The report is also emitted as JSON for "
        "an app UI, and the same lines are written to the ComfyUI log."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "detail": (
                    ["summary", "full"],
                    {
                        "default": "summary",
                        "tooltip": (
                            "How much of the catalog to list. 'summary' shows one recommended file per task plus "
                            "the reason for the choice; 'full' lists every alternative with its size, its stars "
                            "and the measured fit verdict for this machine."
                        ),
                    },
                ),
            },
            "optional": {
                "resources_json": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Optional: a resource snapshot as JSON (the format the diagnostics script writes). "
                            "Connect it to advise for a machine other than the one ComfyUI runs on - for example "
                            "when preparing a workflow for a friend's PC. Empty means 'detect this machine'."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "advice_json")
    FUNCTION = "advise"
    CATEGORY = "Music Production Toolkit/utilities"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Hardware and the model folder can change between runs; a cached answer
        # would be a stale recommendation. NaN never equals itself, so the node
        # always runs.
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # The advisor must never be the reason a queue is refused: it only reports.
        return True

    def advise(self, detail="summary", resources_json=""):
        resources = None
        provided = (resources_json or "").strip() if isinstance(resources_json, str) else ""
        if provided:
            try:
                payload = json.loads(provided)
                resources = _snapshot_from_payload(payload)
            except Exception as exc:
                LOGGER.warning(
                    "Ignoring unreadable resources_json (%s: %s); detecting this machine instead.",
                    type(exc).__name__, exc,
                )
        report = recommend_models(resources=resources)
        lines = format_advisor_lines(report, detail=detail if detail in ("summary", "full") else "summary")
        for line in lines:
            LOGGER.info("%s", line)
        return ("\n".join(lines), json.dumps(report, ensure_ascii=False))


def _snapshot_from_payload(payload: Any) -> Optional[resource_profiles.ResourceSnapshot]:
    """Rebuild a snapshot from the diagnostics JSON (never from a guess)."""
    if not isinstance(payload, dict):
        return None
    source = payload.get("resources") if isinstance(payload.get("resources"), dict) else payload
    if not isinstance(source, dict):
        return None
    if not any(key in source for key in ("devices", "cpu_count", "ram_total_bytes")):
        return None
    devices: List[resource_profiles.DeviceInfo] = []
    for raw in source.get("devices") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        devices.append(
            resource_profiles.DeviceInfo(
                id=str(raw.get("id")),
                name=str(raw.get("name") or raw.get("id")),
                kind=str(raw.get("kind") or "cpu"),
                backend=str(raw.get("backend") or "cpu"),
                logical=raw.get("logical_index"),
                physical=raw.get("physical_index"),
                vram_total_bytes=raw.get("vram_total_bytes"),
                vram_free_bytes=raw.get("vram_free_bytes"),
                supported=bool(raw.get("supported", True)),
            )
        )
    if not devices:
        return None
    if not any(device.kind == "cpu" for device in devices):
        devices.append(
            resource_profiles.DeviceInfo(id="cpu", name="CPU", kind="cpu", backend="cpu")
        )
    return resource_profiles.ResourceSnapshot(
        cpu_count=source.get("cpu_count"),
        ram_total_bytes=source.get("ram_total_bytes"),
        ram_available_bytes=source.get("ram_available_bytes"),
        devices=devices,
        backends=dict(source.get("backends") or {}),
        backend_versions=dict(source.get("backend_versions") or {}),
        visible_devices_env=source.get("visible_devices_env"),
        notes=list(source.get("notes") or []),
        captured_at=str(source.get("captured_at") or ""),
    )


NODE_CLASS_MAPPINGS = {"MiniMaxModelAdvisor": MiniMaxModelAdvisor}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxModelAdvisor": "Model advisor · what suits this machine"}
