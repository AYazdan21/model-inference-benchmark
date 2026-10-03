"""
Checks behind the benchmark sweep (see src/benchmark/sweep.py).

1. Threshold equivalence: running a model once at a low confidence and dropping the boxes below a higher threshold must give
   exactly the boxes of a run at that higher threshold (this is how the sweep gets its confidence values without extra
   inference). Compared per frame for every given model (box coordinates, confidence, class).
2. --self-test: the quality functions (agreement F1, temporal consistency, stability, best-configuration rule) on hand-made cases.

    docker compose -f docker/docker-compose.yml run --rm -T -e YOLO_CONFIG_DIR=/tmp x86-cpu \\
        python scripts/check_sweep_equivalence.py --video videos/input/<file>.mp4 --models models/HumanDetection_light_input_640.onnx
    python scripts/check_sweep_equivalence.py --self-test        # host Python is enough
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import sweep as sw  # noqa: E402


def self_test() -> int:
    failures = []

    def check(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' - ' + detail) if detail and not cond else ''}")
        if not cond:
            failures.append(name)

    box = lambda x, y, w, h, conf=0.9, cls=0: (x, y, x + w, y + h, conf, cls, "person")
    print("agreement / temporal / stability on hand-made cases")
    ref = [[box(.1, .1, .2, .4), box(.6, .1, .2, .4)], [box(.1, .1, .2, .4)]]
    check("identical boxes -> F1 1.0", sw.agreement(ref, ref)["f1"] == 1.0)
    one_missing = [[box(.1, .1, .2, .4)], [box(.1, .1, .2, .4)]]
    a = sw.agreement(ref, one_missing)
    check("one of three reference boxes missing -> precision 1, recall 2/3", a["precision"] == 1.0 and abs(a["recall"] - 0.6667) < 1e-3, str(a))
    extra = [[box(.1, .1, .2, .4), box(.6, .1, .2, .4), box(.4, .6, .1, .2)], [box(.1, .1, .2, .4)]]
    a = sw.agreement(ref, extra)
    check("one extra box -> recall 1, precision 3/4", a["recall"] == 1.0 and abs(a["precision"] - 0.75) < 1e-3, str(a))
    check("different class does not match", sw.agreement(ref, [[box(.1, .1, .2, .4, cls=1), box(.6, .1, .2, .4, cls=1)], [box(.1, .1, .2, .4, cls=1)]])["f1"] == 0.0)
    check("IoU below 0.5 does not match", sw.agreement([[box(0, 0, .2, .2)]], [[box(.12, 0, .2, .2)]])["f1"] == 0.0)
    check("nothing detected by either side -> 1.0", sw.agreement([[], []], [[], []])["f1"] == 1.0)
    check("nothing from the candidate -> 0.0", sw.agreement(ref, [[], []])["f1"] == 0.0)
    check("duplicate box counts as a false positive", abs(sw.agreement([[box(.1, .1, .2, .4)]], [[box(.1, .1, .2, .4, .9), box(.101, .1, .2, .4, .8)]])["precision"] - 0.5) < 1e-6)
    steady = [[box(.1, .1, .2, .4)], [box(.11, .1, .2, .4)], [box(.12, .1, .2, .4)]]
    check("steady track -> temporal 1.0", sw.temporal_consistency(steady) == 1.0)
    flicker = [[box(.1, .1, .2, .4)], [], [box(.1, .1, .2, .4)], []]
    check("flicker pairs: 1st pair lost, 2nd skipped (empty), 3rd lost -> 0.0", sw.temporal_consistency(flicker) == 0.0)
    check("no boxes at all -> temporal n/a", sw.temporal_consistency([[], []]) is None)
    check("stability = mean of both", sw.stability_score(0.8, 0.6) == 0.7)
    check("stability with only one part", sw.stability_score(0.8, None) == 0.8 and sw.stability_score(None, 0.5) == 0.5 and sw.stability_score(None, None) is None)
    d = [[(0, 0, .1, .1, 0.9, 0, "p"), (.5, .5, .6, .6, 0.3, 0, "p")]]
    check("filter_frames keeps >= threshold", len(sw.filter_frames(d, 0.3)[0]) == 2 and len(sw.filter_frames(d, 0.31)[0]) == 1)
    check("pick_sample_frames: most detections, spread", sw.pick_sample_frames([0, 5, 5, 1, 0, 0, 4, 0, 0, 0], 3) == [1, 2, 6])
    check("pick_sample_frames: nothing detected -> evenly spaced", sw.pick_sample_frames([0] * 10, 3) == [0, 4, 9])

    print("sources / inputs / settings")
    srcs = sw.resolve_sources([720, 1080, 0, 2160], 2880, 1616)
    check("heights above native are dropped, native kept once", [s["label"] for s in srcs] == ["native", "1080p", "720p"], str([s["label"] for s in srcs]))
    check("720p width keeps the aspect ratio and is even", srcs[2]["width"] == 1284, str(srcs[2]))
    check("locked model: single input size", sw.resolve_inputs(False, (640, 640), [480, 800]) == [(640, 640)])
    check("dynamic model: requested sizes plus default, largest first", sw.resolve_inputs(True, (640, 640), [480, 800]) == [(800, 800), (640, 640), (480, 480)])
    check("dynamic model, no sizes requested: default only", sw.resolve_inputs(True, (640, 640), []) == [(640, 640)])
    n = sw.normalize_sweep({"conf_thresholds": [0.5, 0.25]}, 0.35)
    check("run confidence is added to the thresholds", n["conf_thresholds"] == [0.25, 0.35, 0.5], str(n))
    try:
        sw.normalize_sweep({"input_sizes": [100]}, 0.35)
        check("input size must be a multiple of 32", False)
    except ValueError:
        check("input size must be a multiple of 32", True)

    print("best-configuration rule")

    def entry(src_h, size, conf, fps, stab, rt_req=25.0, ratings=None, f1=None, tc=None):
        label = "native" if src_h == 1616 else f"{src_h}p"
        return {"id": f"{label}|{size}x{size}|{conf:g}", "label": f"{label} / {size} / {conf:g}", "short": "s", "conf": conf,
                "source": {"label": label, "height": src_h}, "input": {"h": size, "w": size},
                "device_estimate": {"est_fps": fps}, "host_performance": {"throughput_fps": fps},
                "realtime": {"realtime_capable": fps >= rt_req, "required_fps": rt_req}, "stability": stab,
                "agreement": {"f1": f1 if f1 is not None else stab}, "temporal": tc if tc is not None else stab, "ratings": ratings}

    es = [entry(1616, 800, .35, 10, .99), entry(1080, 640, .35, 30, .93), entry(720, 640, .35, 40, .91), entry(720, 480, .35, 60, .80)]
    b = sw.pick_best(es, 0.35)
    check("auto: most stable real-time configuration", b["config_id"] == "1080p|640x640|0.35" and b["rule"] == sw.RULE_AUTO, str(b))
    es2 = [entry(1080, 640, .35, 30, .93), entry(720, 640, .35, 40, .93)]
    check("auto: tie on stability -> higher source resolution", sw.pick_best(es2, 0.35)["config_id"] == "1080p|640x640|0.35")
    es3 = [entry(1080, 640, .35, 5, .9), entry(720, 480, .35, 12, .8)]
    b = sw.pick_best(es3, 0.35)
    check("auto: nothing real-time -> fastest, flagged", b["config_id"] == "720p|480x480|0.35" and b["rule"] == sw.RULE_FASTEST and "not real-time at any tested setting" in b["reason"], str(b))
    check("single configuration", sw.pick_best([es[0]], 0.35)["rule"] == sw.RULE_SINGLE)
    rated = [dict(e) for e in es]
    rated[3]["ratings"] = {"coverage": 4, "duplicates": 2}
    rated[1]["ratings"] = {"coverage": 5, "duplicates": 3}
    b = sw.pick_best(rated, 0.35)
    check("ratings win: highest coverage among rated", b["config_id"] == "1080p|640x640|0.35" and b["rule"] == sw.RULE_USER, str(b))
    rated[2]["ratings"] = {"coverage": 5, "duplicates": 1}
    b = sw.pick_best(rated, 0.35)
    check("ratings: equal coverage -> fewer duplicates", b["config_id"] == "720p|640x640|0.35", str(b))
    rated[1]["ratings"] = {"coverage": 5, "duplicates": None}
    rated[2]["ratings"] = {"coverage": 5, "duplicates": 7}
    check("ratings: blank duplicates rank after entered ones", sw.pick_best(rated, 0.35)["config_id"] == "720p|640x640|0.35")
    ap = sw.pick_best(rated, 0.35)["auto_pick"]
    check("automatic pick is shown for comparison when it differs", ap is not None and ap["config_id"] == "1080p|640x640|0.35", str(ap))
    print("self-test:", "FAILED " + ", ".join(failures) if failures else "all passed")
    return 1 if failures else 0


def box_key(d):
    return (tuple(round(float(c), 3) for c in d.box), round(float(d.confidence), 4), int(d.class_id))


def equivalence(models, video, frames, targets, base_confs, size_hint):
    import copy

    import cv2

    from src.config import load_yaml
    from src.runtimes import get_detector
    from src.video import VideoReader

    cfg = load_yaml(PROJECT_ROOT / "configs" / "detection.yaml")
    bad = 0
    reader = VideoReader(PROJECT_ROOT / video)
    picks = [int(i * max(1, reader.total_frames - 1) / max(1, frames)) for i in range(frames)]
    imgs = []
    for idx in picks:
        reader.seek_frame(idx)
        ok, frame, _ = reader.read_frame()
        if ok:
            h = 720
            imgs.append((idx, cv2.resize(frame, (int(frame.shape[1] * h / frame.shape[0]), h), interpolation=cv2.INTER_AREA)))
    reader.release()
    print(f"{len(imgs)} frames of {video} (indices {[i for i, _ in imgs]}), scaled to 720p")
    for model in models:
        c = copy.deepcopy(cfg)
        det = get_detector("x86-cpu", str(PROJECT_ROOT / model), c)
        name = Path(model).name
        for base in base_confs:
            det.set_conf(base)
            low = [det.predict(img).detections for _, img in imgs]
            print(f"\n{name}: one run at conf {base} ({sum(len(d) for d in low)} boxes in total) vs. runs at higher thresholds")
            for t in targets:
                if t <= base:
                    continue
                det.set_conf(t)
                direct = [det.predict(img).detections for _, img in imgs]
                same, boxes = 0, 0
                for lo, di in zip(low, direct):
                    derived = sorted(box_key(d) for d in lo if d.confidence >= t)
                    got = sorted(box_key(d) for d in di)
                    boxes += len(got)
                    same += derived == got
                ok = same == len(imgs)
                bad += 0 if ok else 1
                print(f"  conf {t}: {same}/{len(imgs)} frames identical, {boxes} boxes  -> {'IDENTICAL' if ok else 'DIFFERENT'}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="only run the hand-made quality-function cases")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=6, help="frames spread over the video")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.25, 0.35, 0.5])
    ap.add_argument("--base-confs", nargs="+", type=float, default=[0.25, 0.1],
                    help="confidence of the single low run the others are derived from")
    args = ap.parse_args()
    if args.self_test or not args.models:
        rc = self_test()
        if not args.models:
            sys.exit(rc)
    if not args.video:
        ap.error("--video is needed with --models")
    bad = equivalence(args.models, args.video, args.frames, args.targets, args.base_confs, (640, 640))
    print("\nRESULT:", "all derived results identical to direct runs" if bad == 0 else f"{bad} mismatch(es)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
