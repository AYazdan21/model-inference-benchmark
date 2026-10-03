"""
Benchmark sweeps: every model / device pair is tested over several configurations and the best one is reported.

A configuration is (source resolution, model input size, confidence threshold):
  * source resolution  height of the frame the "camera" delivers (frames are downscaled outside the timed section)
  * model input size   only for models with a dynamic input (Ultralytics weights, ONNX graphs with symbolic H/W);
                       models with a static input run at their own size ("locked")
  * confidence         NOT swept by inference: the model runs once per (source, input) at the lowest threshold and the
                       higher thresholds are obtained by dropping boxes below them. Greedy NMS is consistent under score
                       filtering (a box is kept iff no higher-scored kept box overlaps it, and all those boxes pass any
                       lower threshold too), so this equals running again at that threshold.

This module has no heavy dependencies: configuration handling, the label-free quality proxies (agreement with a
reference configuration, temporal consistency, stability), the best-configuration rule and the application of manual
ratings to a stored report. The execution engine is in sweep_runner.py.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.benchmark.ratings import RatingStore, conf_text, make_key

DEFAULT_SWEEP: Dict[str, List[Any]] = {
    "source_heights": [720, 1080, 0],      # 0 = native resolution of the video
    "input_sizes": [480, 640, 800],        # dynamic-input models only (plus the model's own default size)
    "conf_thresholds": [0.25, 0.35, 0.5],  # the confidence of the run / UI is always added
}
SOURCE_HEIGHT_CHOICES = (480, 720, 1080, 0)
INPUT_SIZE_CHOICES = (320, 480, 640, 800, 960)
PRESETS: Dict[str, Dict[str, List[Any]]] = {
    "quick": {"source_heights": [720, 0], "input_sizes": [], "conf_thresholds": [0.25, 0.35, 0.5]},
    "full": {"source_heights": [480, 720, 1080, 0], "input_sizes": [320, 480, 640, 800, 960],
             "conf_thresholds": [0.25, 0.35, 0.5]},
}
MAX_VALUES = 8            # per dimension
MAX_CONFIGS = 200
IOU_AGREEMENT = 0.5       # boxes of two configurations are "the same detection" above this IoU (same class)
IOU_TEMPORAL = 0.3        # a box found again in the next frame (same class) above this IoU
STABILITY_DECIMALS = 2    # stability scores that agree to this many decimals count as a tie

RULE_USER, RULE_AUTO, RULE_FASTEST, RULE_SINGLE = "user_rating", "auto_realtime_stable", "auto_fastest", "single_config"
RULE_NAMES = {
    RULE_USER: "manual rating", RULE_AUTO: "real-time + stable (automatic)",
    RULE_FASTEST: "fastest, not real-time (automatic)", RULE_SINGLE: "single configuration",
}

SWEEP_NOTES = [
    "Confidence is swept without extra inference: the model runs once per source resolution and input size at the lowest "
    "threshold and higher thresholds drop boxes below them (greedy NMS gives the same boxes either way). Postprocess time "
    "is therefore measured at the lowest threshold and reused, a slight over-estimate for the higher ones.",
    "Source resolution: every decoded frame is downscaled to the given height (aspect ratio kept) before it is timed; "
    "decoding and downscaling are not part of latency or FPS (a real camera would deliver that resolution).",
    "Model input size is swept only for models with a dynamic input; models with a static input are locked to their own size. "
    "GFLOPs and the device latency estimate are recomputed for every input size.",
    "Agreement F1 compares the boxes of a configuration with the reference configuration (highest source resolution, "
    "largest input size, the run's confidence); the reference scores 1.0 by definition, so confidence values far from the "
    "run's confidence lose agreement by construction. Temporal consistency is the share of boxes that reappear in the next frame.",
    "Peak RAM is the peak of the benchmark process (model loaded once; it can include memory kept from earlier "
    "configurations of the same model; cached video frames of the harness are subtracted).",
]

SELECTION_RULE_STEPS = [
    "If you entered manual ratings for this video and model, only the rated configurations compete: highest coverage first, "
    "then fewest duplicates (blank duplicates rank after entered ones), then real-time on this device, then the stability "
    "score, then the higher source resolution. The automatic pick is shown next to it for comparison when it differs.",
    "Otherwise the real-time configurations (estimated device FPS, host FPS for the host-measured target, at least the required "
    "FPS) compete: the highest stability score wins (scores equal to two decimals tie), then the higher source resolution, "
    "then the larger input size.",
    "If no configuration is real-time the fastest one is shown and flagged \"not real-time at any tested setting\".",
]
STABILITY_FORMULA = "stability = 0.5 x agreement F1 (vs. the reference configuration) + 0.5 x temporal consistency"


# ---------------------------------------------------------------- settings

def height_value(v: Any) -> int:
    """Source height from a UI / CLI value: 720, "720", "720p", "native" or 0 (native). 0 = native resolution."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "native", "full", "0"):
            return 0
        v = s[:-1] if s.endswith("p") else s
    try:
        h = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"bad source height '{v}' (use e.g. 720, 1080 or native)") from None
    if h == 0:
        return 0
    if not 120 <= h <= 4320:
        raise ValueError(f"source height {h} is outside 120..4320")
    return h


def size_value(v: Any) -> int:
    try:
        s = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"bad input size '{v}'") from None
    if s % 32 or not 64 <= s <= 2048:
        raise ValueError(f"input size {s} must be a multiple of 32 between 64 and 2048")
    return s


def conf_value(v: Any) -> float:
    try:
        c = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"bad confidence '{v}'") from None
    if not 0.01 <= c <= 0.99:
        raise ValueError(f"confidence {c} must be between 0.01 and 0.99")
    return round(c, 3)


def _list(raw: Any, conv, what: str) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, (str, int, float)):
        raw = [x for x in str(raw).replace(";", ",").replace(" ", ",").split(",") if x != ""]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{what} must be a list")
    out: List[Any] = []
    for x in raw:
        v = conv(x)
        if v not in out:
            out.append(v)
    if len(out) > MAX_VALUES:
        raise ValueError(f"at most {MAX_VALUES} {what}")
    return out


def normalize_sweep(raw: Optional[Dict[str, Any]], default_conf: float, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validated sweep settings {enabled, source_heights, input_sizes, conf_thresholds}.
    Missing keys come from `base` (the configs/detection.yaml defaults). The run's confidence is always one of the
    thresholds. input_sizes may be empty: dynamic models then run at their own default size only.
    Raises ValueError for invalid values."""
    base = {**DEFAULT_SWEEP, **(base or {})}
    raw = raw or {}
    if raw.get("enabled") is False:
        return {"enabled": False, "source_heights": [0], "input_sizes": [], "conf_thresholds": [conf_value(default_conf)]}
    heights = _list(raw.get("source_heights", base["source_heights"]), height_value, "source heights") or [0]
    sizes = _list(raw.get("input_sizes", base["input_sizes"]), size_value, "input sizes")
    confs = _list(raw.get("conf_thresholds", base["conf_thresholds"]), conf_value, "confidence thresholds")
    default = conf_value(default_conf)
    if default not in confs:
        confs.append(default)
    heights.sort(key=lambda h: (h == 0, h), reverse=True)
    sizes.sort(reverse=True)
    confs.sort()
    if len(confs) > MAX_VALUES:
        raise ValueError(f"at most {MAX_VALUES} confidence thresholds (the run's confidence is added)")
    if len(heights) * max(1, len(sizes)) * len(confs) > MAX_CONFIGS:
        raise ValueError(f"too many configurations (limit {MAX_CONFIGS})")
    return {"enabled": True, "source_heights": heights, "input_sizes": sizes, "conf_thresholds": confs}


def sweep_cli_args(sweep: Optional[Dict[str, Any]]) -> List[str]:
    """Command-line flags of run_benchmark.py for stored sweep settings (None = pre-sweep behaviour)."""
    if not sweep or sweep.get("enabled") is False:
        return ["--no-sweep"]
    return ["--source-heights", *["native" if h == 0 else str(h) for h in sweep["source_heights"]],
            "--input-sizes", *([str(s) for s in sweep["input_sizes"]] or ["default"]),
            "--conf-thresholds", *[conf_text(c) for c in sweep["conf_thresholds"]]]


def count_configs(sweep: Dict[str, Any], dynamic: bool, native_height: Optional[int] = None) -> int:
    heights = len(effective_heights(sweep["source_heights"], native_height)) if native_height else len(sweep["source_heights"])
    sizes = max(1, len(sweep["input_sizes"])) if dynamic else 1
    return heights * sizes * len(sweep["conf_thresholds"])


# ---------------------------------------------------------------- configurations

def effective_heights(heights: Sequence[int], native_height: int) -> List[int]:
    """Requested heights without those at or above the native height (they are the native resolution), 0 = native."""
    out = []
    for h in heights:
        eff = 0 if h == 0 or h >= native_height else h
        if eff not in out:
            out.append(eff)
    return sorted(out, key=lambda h: (h == 0, h), reverse=True)


def resolve_sources(heights: Sequence[int], native_w: int, native_h: int) -> List[Dict[str, Any]]:
    """[{label, text, height, width, native}] highest first. Heights at or above the native one collapse into 'native'."""
    out = []
    for h in effective_heights(heights, native_h):
        if h == 0:
            out.append({"label": "native", "text": f"native ({native_w}x{native_h})", "height": native_h,
                        "width": native_w, "native": True})
        else:
            w = max(2, int(round(native_w * h / native_h / 2.0)) * 2)
            out.append({"label": f"{h}p", "text": f"{h}p ({w}x{h})", "height": h, "width": w, "native": False})
    return out


def resolve_inputs(dynamic: bool, default_size: Tuple[int, int], requested: Sequence[int]) -> List[Tuple[int, int]]:
    """Input sizes (h, w) tested for a model, largest first: only the default for a locked model; for a dynamic one the
    requested square sizes plus the model's default (just the default when none are requested)."""
    default = (int(default_size[0]), int(default_size[1]))
    if not dynamic:
        return [default]
    sizes = {default} | {(int(s), int(s)) for s in requested}
    return sorted(sizes, key=lambda hw: (hw[0] * hw[1], hw), reverse=True)


def input_label(hw: Sequence[int]) -> str:
    return f"{int(hw[0])}x{int(hw[1])}"


def input_short(hw: Sequence[int]) -> str:
    return str(int(hw[0])) if int(hw[0]) == int(hw[1]) else input_label(hw)


def config_id(source_label: str, hw: Sequence[int], conf: float) -> str:
    return f"{source_label}|{input_label(hw)}|{conf_text(conf)}"


def config_slug(cid: str) -> str:
    """File-name-safe form of a config id (used for sample images)."""
    return cid.replace("|", "_")


def config_text(source: Dict[str, Any], hw: Sequence[int], conf: float, locked: bool = False) -> str:
    return f"{source['text']} · input {input_label(hw)}{' (locked)' if locked else ''} · conf {conf_text(conf)}"


def config_short(source: Dict[str, Any], hw: Sequence[int], conf: float) -> str:
    return f"{source['label']} · {input_short(hw)} · {conf_text(conf)}"


# ---------------------------------------------------------------- quality proxies
# A detection is (x1, y1, x2, y2, confidence, class_id, class_name) in coordinates normalised by the frame size, so
# configurations with different source resolutions can be compared.

def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def filter_frames(frames: Sequence[Sequence[Tuple]], conf: float) -> List[List[Tuple]]:
    """Detections of each frame with confidence >= conf (how a higher threshold is derived from a lower one)."""
    return [[d for d in dets if d[4] >= conf] for dets in frames]


def match_count(cand: Sequence[Tuple], ref: Sequence[Tuple], thr: float = IOU_AGREEMENT) -> int:
    """Greedy one-to-one matching of candidate boxes (highest confidence first) to reference boxes of the same class."""
    used = [False] * len(ref)
    tp = 0
    for c in sorted(cand, key=lambda d: -d[4]):
        best, best_iou = -1, thr
        for j, r in enumerate(ref):
            if used[j] or r[5] != c[5]:
                continue
            v = iou(c, r)
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            used[best] = True
            tp += 1
    return tp


def agreement(ref_frames: Sequence[Sequence[Tuple]], cand_frames: Sequence[Sequence[Tuple]],
              thr: float = IOU_AGREEMENT) -> Dict[str, float]:
    """Precision / recall / F1 of a configuration's boxes against the reference's, summed over all frames.
    Nothing detected by either side counts as perfect agreement (1.0)."""
    tp = fp = fn = 0
    for ref, cand in zip(ref_frames, cand_frames):
        m = match_count(cand, ref, thr)
        tp += m
        fp += len(cand) - m
        fn += len(ref) - m
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def temporal_consistency(frames: Sequence[Sequence[Tuple]], thr: float = IOU_TEMPORAL) -> Optional[float]:
    """Mean over consecutive frame pairs of the share of boxes in frame t that have a same-class box (IoU >= thr) in
    frame t+1; pairs whose first frame has no box are skipped. None when there is nothing to measure."""
    shares = []
    for cur, nxt in zip(frames[:-1], frames[1:]):
        if not cur:
            continue
        hit = sum(1 for c in cur if any(n[5] == c[5] and iou(c, n) >= thr for n in nxt))
        shares.append(hit / len(cur))
    return round(sum(shares) / len(shares), 4) if shares else None


def stability_score(f1: Optional[float], temporal: Optional[float]) -> Optional[float]:
    """0.5 x agreement F1 + 0.5 x temporal consistency (whichever exists when one is not measurable)."""
    if f1 is None and temporal is None:
        return None
    if f1 is None:
        return temporal
    if temporal is None:
        return round(f1, 4)
    return round(0.5 * f1 + 0.5 * temporal, 4)


def pick_sample_frames(counts: Sequence[int], k: int) -> List[int]:
    """Indices (ascending) of up to k frames with the most detections, spread out when possible; evenly spaced when
    nothing was detected anywhere."""
    n = len(counts)
    k = min(k, n)
    if k <= 0:
        return []
    if max(counts, default=0) == 0:
        return sorted({int(round(i * (n - 1) / max(1, k - 1))) for i in range(k)}) if k > 1 else [0]
    gap = max(1, n // (2 * k))
    order = sorted(range(n), key=lambda i: (-counts[i], i))
    picked: List[int] = []
    for i in order:
        if len(picked) < k and all(abs(i - p) >= gap for p in picked):
            picked.append(i)
    for i in order:  # not enough frames far apart: fill with the next best
        if len(picked) >= k:
            break
        if i not in picked:
            picked.append(i)
    return sorted(picked)


# ---------------------------------------------------------------- best configuration

def entry_fps(entry: Dict[str, Any]) -> float:
    est = entry.get("device_estimate")
    return float(est["est_fps"] if est else entry["host_performance"]["throughput_fps"])


def _is_realtime(entry: Dict[str, Any]) -> bool:
    return bool(entry["realtime"]["realtime_capable"])


def _stab(entry: Dict[str, Any]) -> float:
    s = entry.get("stability")
    return round(s, STABILITY_DECIMALS) if s is not None else -1.0


def _res_key(entry: Dict[str, Any]) -> Tuple[int, int]:
    return entry["source"]["height"], entry["input"]["h"] * entry["input"]["w"]


def _stability_text(entry: Dict[str, Any]) -> str:
    s, ag, tc = entry.get("stability"), entry.get("agreement"), entry.get("temporal")
    if s is None:
        return "stability not measurable"
    parts = []
    if ag is not None:
        parts.append(f"agreement F1 {ag['f1']:.2f}")
    if tc is not None:
        parts.append(f"temporal {tc:.2f}")
    return f"stability {s:.2f}" + (f" ({', '.join(parts)})" if parts else "")


def _rt_text(entry: Dict[str, Any]) -> str:
    rt = entry["realtime"]
    if rt["realtime_capable"]:
        return f"real-time at {entry_fps(entry):.1f} FPS ({rt['required_fps']:g} needed)"
    return f"not real-time at {entry_fps(entry):.1f} FPS ({rt['required_fps']:g} needed)"


def pick_auto(entries: Sequence[Dict[str, Any]], default_conf: float) -> Tuple[Dict[str, Any], str, str]:
    """(entry, rule, reason) of the automatic real-time + stable choice."""
    if len(entries) == 1:
        e = entries[0]
        return e, RULE_SINGLE, f"only one configuration was tested; {_rt_text(e)}"
    rt = [e for e in entries if _is_realtime(e)]
    if rt:
        best = max(rt, key=lambda e: (_stab(e), *_res_key(e), -abs(e["conf"] - default_conf), -e["conf"]))
        if best.get("stability") is None:
            why = ("output not decoded by this harness, so quality cannot be measured: the highest real-time "
                   "resolution is used")
        else:
            why = f"most stable of {len(rt)} real-time configuration(s): {_stability_text(best)}"
        return best, RULE_AUTO, f"{_rt_text(best)}; {why}"
    best = max(entries, key=lambda e: (entry_fps(e), _stab(e), *_res_key(e)))
    return best, RULE_FASTEST, f"not real-time at any tested setting; the fastest is shown, at {entry_fps(best):.1f} FPS ({best['realtime']['required_fps']:g} needed)"


def _rating_text(r: Dict[str, Any]) -> str:
    cov = f"coverage {r['coverage']}/5" if r.get("coverage") is not None else "coverage not entered"
    dup = f"{r['duplicates']} duplicates" if r.get("duplicates") is not None else "duplicates not entered"
    return f"{cov}, {dup}"


def pick_best(entries: Sequence[Dict[str, Any]], default_conf: float) -> Dict[str, Any]:
    """Best configuration of one model/device pair (entries carry their `ratings`): see SELECTION_RULE_STEPS."""
    auto, auto_rule, auto_reason = pick_auto(entries, default_conf)
    rated = [e for e in entries if e.get("ratings")]
    if not rated:
        return {"config_id": auto["id"], "label": auto["label"], "rule": auto_rule, "reason": auto_reason,
                "rated": False, "rated_count": 0, "auto_pick": None}

    def key(e: Dict[str, Any]):
        r = e["ratings"]
        return (r["coverage"] if r.get("coverage") is not None else -1,
                -(r["duplicates"] if r.get("duplicates") is not None else 10 ** 9),
                1 if _is_realtime(e) else 0, _stab(e), *_res_key(e), -abs(e["conf"] - default_conf))

    best = max(rated, key=key)
    reason = (f"user rating: {_rating_text(best['ratings'])}; {_rt_text(best)}"
              + (f"; best of {len(rated)} rated configurations" if len(rated) > 1 else ""))
    auto_pick = None
    if auto["id"] != best["id"]:
        auto_pick = {"config_id": auto["id"], "label": auto["label"], "rule": auto_rule, "reason": auto_reason}
    return {"config_id": best["id"], "label": best["label"], "rule": RULE_USER, "reason": reason, "rated": True,
            "rated_count": len(rated), "auto_pick": auto_pick}


# ---------------------------------------------------------------- report integration

def ratings_signature(entries: Sequence[Dict[str, Any]]) -> str:
    items = sorted((e["id"], (e.get("ratings") or {}).get("coverage"), (e.get("ratings") or {}).get("duplicates"))
                   for e in entries if e.get("ratings"))
    return hashlib.sha1(json.dumps(items).encode("utf-8")).hexdigest()[:12]


def find_entry(report: Dict[str, Any], config_id: str) -> Dict[str, Any]:
    for e in report["sweep"]["configs"]:
        if e["id"] == config_id:
            return e
    raise KeyError(config_id)


def best_entry(report: Dict[str, Any]) -> Dict[str, Any]:
    return find_entry(report, report["sweep"]["best"]["config_id"])


def overlay_entry(report: Dict[str, Any], entry: Dict[str, Any]) -> None:
    """Makes the top-level sections of a report describe one configuration. Nested dicts are replaced, never edited,
    so a shallow copy of the report can be overlaid without touching the original."""
    sw = report["sweep"]
    report["host_performance"] = entry["host_performance"]
    report["device_estimate"] = entry["device_estimate"]
    report["realtime"] = entry["realtime"]
    report["resources"] = entry["resources"]
    report["detections"] = entry["detections"]
    best = sw.get("best") or {}
    info = {"level": "ok",
            "message": f"Configuration shown: {entry['label']}"
                       + (f" ({best['reason']})" if best.get("config_id") == entry["id"] and best.get("reason") else ".")}
    report["verdict"] = {"overall": entry["verdict"]["overall"], "items": [info] + list(entry["verdict"]["items"])}
    report["config"] = {**report["config"], "conf_threshold": entry["conf"],
                        "default_conf_threshold": sw["dimensions"]["default_conf"],
                        "source_resolution": {k: entry["source"][k] for k in ("label", "text", "width", "height")},
                        "input_size": [entry["input"]["h"], entry["input"]["w"]]}
    model = {**report["model"], "input_size": [entry["input"]["h"], entry["input"]["w"]]}
    if entry.get("gflops") is not None:
        model["gflops"] = entry["gflops"]
    if entry.get("params_m") is not None:
        model["params_m"] = entry["params_m"]
    report["model"] = model
    report["samples"] = list(entry.get("samples") or [])


def apply_ratings(report: Dict[str, Any], store: Optional[RatingStore] = None,
                  ratings: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
    """Attaches the stored ratings of the report's video and model to its configurations, picks the best configuration
    again and makes the top-level sections follow it. Returns True when anything visible changed."""
    sw = report["sweep"]
    ratings = (store or RatingStore()).load() if ratings is None else ratings
    video, model = sw.get("video") or "synthetic", report["model"]["file"]
    for e in sw["configs"]:
        r = ratings.get(make_key(video, model, e["source"]["label"], input_label((e["input"]["h"], e["input"]["w"])), e["conf"]))
        e["ratings"] = {"coverage": r.get("coverage"), "duplicates": r.get("duplicates"), "updated": r.get("updated")} if r else None
    before = (sw.get("best") or {}).get("config_id"), (sw.get("best") or {}).get("reason"), sw.get("ratings_signature")
    sw["best"] = pick_best(sw["configs"], sw["dimensions"]["default_conf"])
    sw["ratings_signature"] = ratings_signature(sw["configs"])
    overlay_entry(report, best_entry(report))
    return before != (sw["best"]["config_id"], sw["best"]["reason"], sw["ratings_signature"])


def report_view(report: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow copy of a sweep report whose top-level sections describe `entry` (for per-configuration table rows)."""
    view = dict(report)
    overlay_entry(view, entry)
    return view


def rating_summary(entry: Dict[str, Any]) -> str:
    r = entry.get("ratings")
    if not r:
        return "not rated"
    return _rating_text(r)


def best_summary(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compact description of the best configuration for lists and manifests (None for a report without sweep)."""
    sw = report.get("sweep")
    if not sw:
        return None
    e, b = best_entry(report), sw["best"]
    return {
        "config_id": e["id"], "label": e["label"], "short": e["short"], "source": e["source"]["label"],
        "input": input_label((e["input"]["h"], e["input"]["w"])), "conf": e["conf"], "rule": b["rule"], "reason": b["reason"],
        "rated": b["rated"], "auto_pick": (b.get("auto_pick") or {}).get("label"), "n_configs": len(sw["configs"]),
        "stability": e.get("stability"), "agreement_f1": (e.get("agreement") or {}).get("f1"),
        "temporal": e.get("temporal"), "coverage": (e.get("ratings") or {}).get("coverage"),
        "duplicates": (e.get("ratings") or {}).get("duplicates"),
    }
