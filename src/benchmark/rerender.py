"""
Re-rendering of stored benchmark reports and suites after the manual ratings changed (no model is run again).

A rating belongs to (video, model, configuration), so one change can affect several reports and suites: this module finds
every sweep report / suite of that video and model and rebuilds it from its stored data (report.json, frames.csv, suite.json).
Runs on the host: no ONNX / PyTorch needed.
"""
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.benchmark.ratings import RatingStore
from src.benchmark.suite import SUITE_ID_RE, load_manifest, render_suite, save_manifest
from src.utils.logger import get_logger

logger = get_logger("Rerender")
REPORTS_DIR = Path(__file__).resolve().parents[2] / "results" / "reports"


def sweep_report_folders(reports_dir: Optional[Path] = None):
    """(report id, folder, {video, model}) of every single-run bundle with sweep data (suite run folders are not listed)."""
    root = reports_dir or REPORTS_DIR
    if not root.is_dir():
        return
    for folder in sorted(root.iterdir()):
        path = folder / "report.json"
        if not path.is_file() or SUITE_ID_RE.match(folder.name):
            continue
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
            if r.get("sweep"):
                yield folder.name, folder, {"video": r["sweep"].get("video") or "synthetic", "model": r["model"]["file"]}
        except (OSError, ValueError, KeyError):
            continue


def sweep_suite_folders(reports_dir: Optional[Path] = None):
    """(suite id, folder, manifest) of every suite that was run with a sweep."""
    root = reports_dir or REPORTS_DIR
    if not root.is_dir():
        return
    for folder in sorted(root.iterdir()):
        if SUITE_ID_RE.match(folder.name) and (folder / "suite.json").is_file():
            try:
                m = load_manifest(folder)
            except (OSError, ValueError):
                continue
            if (m.get("config") or {}).get("sweep"):
                yield folder.name, folder, m


def suite_video(manifest: Dict[str, Any]) -> str:
    v = (manifest.get("config") or {}).get("video")
    return Path(v).name if v else "synthetic"


def rerender_report(folder: Path, store: Optional[RatingStore] = None, force: bool = False) -> bool:
    """Re-applies the ratings to one report bundle; True when the bundle was rewritten."""
    from src.benchmark.export import rerender_bundle
    path = folder / "report.json"
    before = path.stat().st_mtime_ns
    rerender_bundle(folder, store, force)
    return path.stat().st_mtime_ns != before


def rerender_suite(folder: Path, store: Optional[RatingStore] = None) -> None:
    """Rebuilds a suite: its per-run reports (new best configurations), the manifest cells, exports and the site."""
    manifest = load_manifest(folder)
    render_suite(folder, manifest)
    save_manifest(folder, manifest)


def rerender_matching(pairs: Iterable[Tuple[str, str]], store: Optional[RatingStore] = None, skip_suites: Iterable[str] = (),
                      reports_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """Re-renders every sweep report and suite of the given (video, model) pairs. Suites in skip_suites (a running suite,
    whose runner owns its files) are left alone; they pick the ratings up at their next render."""
    wanted: Set[Tuple[str, str]] = set(pairs)
    skip = set(skip_suites)
    done: Dict[str, List[str]] = {"reports": [], "suites": [], "skipped": []}
    for rid, folder, key in sweep_report_folders(reports_dir):
        if (key["video"], key["model"]) in wanted:
            try:
                rerender_report(folder, store)
                done["reports"].append(rid)
            except (OSError, ValueError, KeyError) as e:
                logger.warning(f"Could not re-render report {rid}: {e}")
    for sid, folder, m in sweep_suite_folders(reports_dir):
        if suite_video(m) in {v for v, _ in wanted} and any((suite_video(m), mod) in wanted for mod in m["config"]["models"]):
            if sid in skip:
                done["skipped"].append(sid)
                continue
            try:
                rerender_suite(folder, store)
                done["suites"].append(sid)
            except (OSError, ValueError, KeyError) as e:
                logger.warning(f"Could not re-render suite {sid}: {e}")
    return done
