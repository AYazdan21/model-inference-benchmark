"""
Manual quality ratings of benchmark configurations, and re-rendering of the reports that depend on them.

A benchmark sweep tests every model over source resolutions x model input sizes x confidence thresholds. You judge the sample
frames of each configuration and enter two values per configuration:
    coverage    1-5   5 = every person is detected, including people far back or partly covered; 1 = many are missed
    duplicates  >= 0  number of duplicate boxes on the same person that you saw (lower is better)
Ratings are stored per (video, model, configuration) in results/ratings/ratings.json and take priority over the automatic
real-time + stable rule when the best configuration of a model/device pair is picked. They apply to every report and suite of
that video and model, on any device. The web app does the same from "Rate detection quality"; this is the command-line way.

    python scripts/rate_configs.py show --report <report_id>            # configuration ids of a report and their ratings
    python scripts/rate_configs.py show --suite <suite_id>
    python scripts/rate_configs.py set --video v.mp4 --model m.onnx --source 720p --input 640x640 --conf 0.35 --coverage 5 --duplicates 0
    python scripts/rate_configs.py set ... --clear                      # remove one rating
    python scripts/rate_configs.py import ratings.json                  # {"ratings": [{video, model, source, input, conf, coverage, duplicates}]}
    python scripts/rate_configs.py list [--video v.mp4] [--model m.onnx]
    python scripts/rate_configs.py clear [--video v.mp4] [--model m.onnx]   # without filters: all ratings
    python scripts/rate_configs.py rerender --report <id> | --suite <id> | --all

set / import / clear re-render the affected reports and suites automatically (--no-rerender to skip). Runs on the host Python.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import sweep as sweep_mod  # noqa: E402
from src.benchmark.ratings import RatingStore, import_entries, make_key  # noqa: E402
from src.benchmark.rerender import (REPORTS_DIR, rerender_matching, rerender_report, rerender_suite,  # noqa: E402
                                    suite_video, sweep_report_folders, sweep_suite_folders)
from src.benchmark.suite import SUITE_ID_RE, load_manifest  # noqa: E402


def print_done(done):
    print(f"Re-rendered {len(done['reports'])} report(s) and {len(done['suites'])} suite(s)"
          + (f" (skipped, still running: {', '.join(done['skipped'])})" if done["skipped"] else ""))


def cmd_list(args, store):
    ratings = store.load()
    rows = [(k, v) for k, v in sorted(ratings.items()) if (not args.video or k.split("|")[0] == args.video) and (not args.model or k.split("|")[1] == args.model)]
    if not rows:
        print("No ratings.")
        return
    print(f"{'video':<40} {'model':<34} {'source':<8} {'input':<9} {'conf':<5} cov dup  updated")
    for k, v in rows:
        video, model, src, inp, conf = k.split("|")
        print(f"{video[:40]:<40} {model[:34]:<34} {src:<8} {inp:<9} {conf:<5} {v.get('coverage') if v.get('coverage') is not None else '-':>3} "
              f"{v.get('duplicates') if v.get('duplicates') is not None else '-':>3}  {v.get('updated') or ''}")


def cmd_show(args, store):
    reports = []
    if args.report:
        for rid, folder, _ in sweep_report_folders():
            if rid == args.report:
                reports.append((folder, json.loads((folder / "report.json").read_text(encoding="utf-8"))))
    elif args.suite:
        if not SUITE_ID_RE.match(args.suite):
            sys.exit("bad suite id")
        folder = REPORTS_DIR / args.suite
        m = load_manifest(folder)
        seen = set()
        for model, row in m["cells"].items():
            for t, c in row.items():
                if c.get("status") == "ok" and c.get("sweep") and model not in seen:
                    seen.add(model)
                    reports.append((folder / "runs" / c["report_id"], json.loads((folder / "runs" / c["report_id"] / "report.json").read_text(encoding="utf-8"))))
    if not reports:
        sys.exit("No sweep report found (reports made before sweeps have a single configuration and cannot be rated).")
    for folder, r in reports:
        sw = r["sweep"]
        print(f"\n{r['model']['file']} | video {sw['video']} | best: {sw['best']['label']} [{sw['best']['rule']}]")
        print(f"  {'source':<8} {'input':<9} {'conf':<5} {'est/host FPS':>12} {'stability':>9}  cov dup")
        for e in sorted(sw["configs"], key=lambda e: (-e["source"]["height"], -(e["input"]["h"] * e["input"]["w"]), e["conf"])):
            rt = e.get("ratings") or {}
            print(f"  {e['source']['label']:<8} {e['input']['label']:<9} {sweep_mod.conf_text(e['conf']):<5} {sweep_mod.entry_fps(e):>12.1f} "
                  f"{(e.get('stability') if e.get('stability') is not None else float('nan')):>9.3f}  {rt.get('coverage') or '-':>3} {rt.get('duplicates') if rt.get('duplicates') is not None else '-':>3}")
        print(f"  rating key example: {make_key(sw['video'], r['model']['file'], sw['configs'][0]['source']['label'], sw['configs'][0]['input']['label'], sw['configs'][0]['conf'])}")


def entry_from_args(args):
    return {"video": args.video, "model": args.model, "source": args.source, "input": args.input, "conf": args.conf,
            "coverage": None if args.clear else args.coverage, "duplicates": None if args.clear else args.duplicates}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--video"); p.add_argument("--model")
    p = sub.add_parser("show"); p.add_argument("--report"); p.add_argument("--suite")
    p = sub.add_parser("set")
    for a in ("video", "model", "source", "input"):
        p.add_argument(f"--{a}", required=True)
    p.add_argument("--conf", required=True)
    p.add_argument("--coverage", type=int); p.add_argument("--duplicates", type=int)
    p.add_argument("--clear", action="store_true", help="remove this configuration's rating")
    p.add_argument("--no-rerender", action="store_true")
    p = sub.add_parser("import"); p.add_argument("file"); p.add_argument("--no-rerender", action="store_true")
    p = sub.add_parser("clear"); p.add_argument("--video"); p.add_argument("--model"); p.add_argument("--no-rerender", action="store_true")
    p = sub.add_parser("rerender")
    p.add_argument("--report"); p.add_argument("--suite"); p.add_argument("--all", action="store_true")
    args = ap.parse_args()
    store = RatingStore()

    if args.cmd == "list":
        cmd_list(args, store)
    elif args.cmd == "show":
        cmd_show(args, store)
    elif args.cmd in ("set", "import", "clear"):
        before = store.load()
        try:
            if args.cmd == "set":
                entries = [entry_from_args(args)]
                n = store.update_many(entries)
                touched = {(args.video, args.model)}
            elif args.cmd == "import":
                entries = import_entries(json.loads(Path(args.file).read_text(encoding="utf-8-sig")))
                n = store.update_many(entries)
                touched = {(e["video"], e["model"]) for e in entries}
            else:
                touched = {tuple(k.split("|")[:2]) for k in before if (not args.video or k.split("|")[0] == args.video) and (not args.model or k.split("|")[1] == args.model)}
                n = store.clear(args.video, args.model)
        except (ValueError, OSError) as e:
            sys.exit(f"Error: {e}")
        print(f"{n} rating(s) changed.")
        if n and not args.no_rerender:
            print_done(rerender_matching(touched, store))
    elif args.cmd == "rerender":
        if args.report:
            folder = REPORTS_DIR / args.report
            if not (folder / "report.json").is_file():
                sys.exit("No such report")
            print("rewritten" if rerender_report(folder, store, force=True) else "no sweep data / unchanged")
        elif args.suite:
            if not SUITE_ID_RE.match(args.suite) or not (REPORTS_DIR / args.suite / "suite.json").is_file():
                sys.exit("No such suite")
            rerender_suite(REPORTS_DIR / args.suite, store)
            print("suite re-rendered")
        elif args.all:
            pairs = {(k["video"], k["model"]) for _, _, k in sweep_report_folders()}
            pairs |= {(suite_video(m), mod) for _, _, m in sweep_suite_folders() for mod in m["config"]["models"]}
            print_done(rerender_matching(pairs, store))
        else:
            ap.error("rerender needs --report, --suite or --all")


if __name__ == "__main__":
    main()
