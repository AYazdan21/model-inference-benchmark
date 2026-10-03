"""
Manual quality ratings of benchmark configurations (user-entered, after a run).

A configuration is (source resolution, model input size, confidence threshold). The user rates, per video and model:
    coverage    1-5   "5 = every person is detected, including people far back or partly covered; 1 = many missed"
    duplicates  >= 0  number of duplicate boxes seen on the same person in the sample frames (lower is better)
Both are optional (blank = not rated). Detections do not depend on the simulated device, so a rating belongs to
(video, model, configuration) and carries over to every report and suite of that video and model, on any device.

Store: results/ratings/ratings.json (UTF-8, written atomically)
    {"version": 1, "ratings": {"<video file>|<model file>|<source res>|<input HxW>|<conf>":
                               {"coverage": 5, "duplicates": 0, "updated": "2026-10-03T12:00:00+03:00"}}}
Examples of the key parts: source res "720p" / "native", input "640x640", conf "0.35". A video-less run uses "synthetic".
"""
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = PROJECT_ROOT / "results" / "ratings" / "ratings.json"
COVERAGE_RANGE = (1, 5)
DUPLICATES_MAX = 9999
_LOCK = threading.RLock()

COVERAGE_HELP = ("5 = every person is detected, including people far back or partly covered; 1 = many people are missed")
DUPLICATES_HELP = "number of duplicate boxes on the same person that you saw in the sample frames (lower is better)"


def conf_text(conf: float) -> str:
    return f"{float(conf):g}"


def make_key(video: str, model: str, source: str, input_label: str, conf: float | str) -> str:
    parts = [str(video or "synthetic"), str(model), str(source), str(input_label), conf if isinstance(conf, str) else conf_text(conf)]
    if any("|" in p or not p for p in parts):
        raise ValueError("rating key parts must be non-empty and must not contain '|'")
    return "|".join(parts)


def clean_value(coverage: Any, duplicates: Any) -> Dict[str, Optional[int]]:
    """Validated {coverage, duplicates}; blank / None means not rated. Raises ValueError for bad input."""
    def to_int(v: Any, name: str) -> Optional[int]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        if isinstance(v, bool):
            raise ValueError(f"{name} must be a whole number")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a whole number") from None
        if f != int(f):
            raise ValueError(f"{name} must be a whole number")
        return int(f)

    cov, dup = to_int(coverage, "coverage"), to_int(duplicates, "duplicates")
    if cov is not None and not COVERAGE_RANGE[0] <= cov <= COVERAGE_RANGE[1]:
        raise ValueError(f"coverage must be between {COVERAGE_RANGE[0]} and {COVERAGE_RANGE[1]}")
    if dup is not None and not 0 <= dup <= DUPLICATES_MAX:
        raise ValueError(f"duplicates must be between 0 and {DUPLICATES_MAX}")
    return {"coverage": cov, "duplicates": dup}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(10):  # Windows: replace fails while a reader has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


class RatingStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else RATINGS_PATH

    # ---- reading
    def load(self) -> Dict[str, Dict[str, Any]]:
        """{key: {coverage, duplicates, updated}}; a missing or unreadable file is an empty store."""
        with _LOCK:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                return {}
        raw = data.get("ratings") if isinstance(data, dict) else None
        out: Dict[str, Dict[str, Any]] = {}
        for key, val in (raw or {}).items():
            if isinstance(val, dict):
                try:
                    clean = clean_value(val.get("coverage"), val.get("duplicates"))
                except ValueError:
                    continue
                if clean["coverage"] is not None or clean["duplicates"] is not None:
                    out[str(key)] = {**clean, "updated": val.get("updated")}
        return out

    def lookup(self, video: str, model: str, source: str, input_label: str, conf: float,
               ratings: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        ratings = self.load() if ratings is None else ratings
        return ratings.get(make_key(video, model, source, input_label, conf))

    def for_model(self, video: str, model: str) -> Dict[str, Dict[str, Any]]:
        """{'<source>|<input>|<conf>': rating} of one video and model."""
        prefix = f"{video or 'synthetic'}|{model}|"
        return {k[len(prefix):]: v for k, v in self.load().items() if k.startswith(prefix)}

    # ---- writing
    def _write(self, ratings: Dict[str, Dict[str, Any]]) -> None:
        _atomic_write(self.path, json.dumps({"version": 1, "ratings": dict(sorted(ratings.items()))},
                                            indent=2, ensure_ascii=False))

    def update_many(self, entries: Iterable[Dict[str, Any]]) -> int:
        """entries: {video, model, source, input, conf, coverage, duplicates}. An entry without a value removes the rating.
        Returns the number of ratings that changed. Raises ValueError (nothing is written) for an invalid entry."""
        prepared = []
        for e in entries:
            key = make_key(e.get("video"), e.get("model"), e.get("source"), e.get("input"), e.get("conf"))
            prepared.append((key, clean_value(e.get("coverage"), e.get("duplicates"))))
        with _LOCK:
            ratings = self.load()
            changed = 0
            for key, val in prepared:
                if val["coverage"] is None and val["duplicates"] is None:
                    if ratings.pop(key, None) is not None:
                        changed += 1
                    continue
                old = ratings.get(key)
                if not old or old.get("coverage") != val["coverage"] or old.get("duplicates") != val["duplicates"]:
                    ratings[key] = {**val, "updated": _now()}
                    changed += 1
            if changed:
                self._write(ratings)
            return changed

    def clear(self, video: Optional[str] = None, model: Optional[str] = None) -> int:
        """Removes the ratings of a video and/or model (all of them when both are None). Returns how many were removed."""
        with _LOCK:
            ratings = self.load()
            keep = {}
            for key, val in ratings.items():
                v, m = key.split("|")[:2]
                if (video is None or v == video) and (model is None or m == model):
                    continue
                keep[key] = val
            removed = len(ratings) - len(keep)
            if removed:
                if keep:
                    self._write(keep)
                else:
                    self._write({})
            return removed


def import_entries(raw: Any) -> List[Dict[str, Any]]:
    """Entries of a ratings import file: the store format, {"ratings": [..]} or a plain list of entry objects.
    Raises ValueError for anything else."""
    if isinstance(raw, dict) and isinstance(raw.get("ratings"), dict):  # the store's own format
        out = []
        for key, val in raw["ratings"].items():
            parts = str(key).split("|")
            if len(parts) != 5 or not isinstance(val, dict):
                raise ValueError(f"bad rating key '{key}'")
            out.append({"video": parts[0], "model": parts[1], "source": parts[2], "input": parts[3], "conf": parts[4],
                        "coverage": val.get("coverage"), "duplicates": val.get("duplicates")})
        return out
    if isinstance(raw, dict) and isinstance(raw.get("ratings"), list):
        raw = raw["ratings"]
    if isinstance(raw, list) and all(isinstance(e, dict) for e in raw):
        return raw
    raise ValueError("expected {\"ratings\": [ {video, model, source, input, conf, coverage, duplicates}, ... ]}")
