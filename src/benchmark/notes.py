"""
User notes per (model, target) pair of a benchmark.

A note is plain text (any Unicode, newlines kept). It is attached to the pair "model file name + target key" and
shows up as a Notes column in the reports. Notes can be handed to the scripts as a UTF-8 JSON file instead of
command-line text (quotes, newlines and non-ASCII text are fragile in `docker compose run ... --notes "..."`):

    {"Gun_Detection_input_640.onnx": {"rk3588-npu": "light model for the entrance camera", "jetson": "..."},
     "other.onnx": "a plain string applies to every target"}
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

NOTES_MAX_CHARS = 2000
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_note(value: Any) -> str:
    """Plain-text note: str, \\n newlines, no control characters, stripped, at most NOTES_MAX_CHARS characters."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = text.encode("utf-8", errors="replace").decode("utf-8")  # lone surrogates cannot be written as UTF-8
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text).strip()
    return text[:NOTES_MAX_CHARS].rstrip()


def clean_notes_map(raw: Any) -> Dict[str, Dict[str, str]]:
    """{model file name: {target: note}} with cleaned, non-empty notes only. A plain string value applies to every
    target and is stored under the key "*". Raises ValueError for a structure that is not a notes map."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("notes must be an object {model: {target: text}}")
    out: Dict[str, Dict[str, str]] = {}
    for model, per_target in raw.items():
        name = Path(str(model)).name
        if isinstance(per_target, str):
            per_target = {"*": per_target}
        if not isinstance(per_target, dict):
            raise ValueError(f"notes for '{model}' must be an object {{target: text}} or a string")
        for target, text in per_target.items():
            if text is not None and not isinstance(text, (str, int, float)):
                raise ValueError(f"note for '{model}' / '{target}' must be text")
            note = clean_note(text)
            if note:
                out.setdefault(name, {})[str(target)] = note
    return out


def load_notes_file(path: str | Path) -> Dict[str, Dict[str, str]]:
    """Reads a notes JSON file (UTF-8, a BOM is tolerated). Raises OSError / ValueError when unreadable or malformed."""
    return clean_notes_map(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def note_for(notes: Dict[str, Dict[str, str]], model: str, target: str) -> str:
    """Note of a pair from a cleaned notes map ("" when there is none)."""
    per_target = notes.get(Path(model).name) or {}
    return per_target.get(target) or per_target.get("*") or ""


def one_line(note: str, limit: Optional[int] = None) -> str:
    """Single-line form for console tables: newlines become ' | ', long text ends with an ellipsis."""
    flat = " | ".join(part.strip() for part in note.split("\n") if part.strip())
    if limit and len(flat) > limit:
        flat = flat[:max(1, limit - 1)].rstrip() + "…"
    return flat


def console_safe(text: str, stream: Any = None) -> str:
    """Text the stream's encoding can print (a Windows console may be cp1252): unprintable characters become '?'."""
    import sys
    enc = getattr(stream or sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")
