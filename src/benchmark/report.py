import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.benchmark.notes import console_safe, one_line

def save_benchmark_report(metrics: Dict[str, Any], output_dir: str | Path = "results/metrics") -> tuple[Path, Path]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = metrics.get("target", "unknown")
    base_name = f"benchmark_{target}_{timestamp}"

    # JSON export
    json_path = out_path / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # CSV summary row export
    sys_res = metrics.get("system_resources", {})
    csv_path = out_path / f"{base_name}.csv"
    row = {
        "timestamp": timestamp,
        "target": target,
        "model": metrics.get("model", ""),
        "notes": metrics.get("notes", ""),
        **_best_columns(metrics),
        "frames": metrics.get("frames_processed", 0),
        "fps": metrics.get("throughput_fps", 0.0),
        "latency_mean_ms": metrics.get("latency_ms", {}).get("mean", 0.0),
        "latency_p50_ms": metrics.get("latency_ms", {}).get("p50", 0.0),
        "latency_p95_ms": metrics.get("latency_ms", {}).get("p95", 0.0),
        "inference_mean_ms": metrics.get("inference_only_ms", {}).get("mean", 0.0),
        "avg_cpu_percent": sys_res.get("avg_cpu_percent", 0.0),
        "ram_limit_mb": sys_res.get("ram_limit_mb", "N/A"),
        "peak_ram_mb": sys_res.get("peak_ram_mb", 0.0),
        "ram_breached": sys_res.get("ram_limit_breached", False),
        "vram_limit_mb": sys_res.get("vram_limit_mb", "N/A"),
        "peak_vram_mb": sys_res.get("peak_vram_mb", 0.0),
        "vram_breached": sys_res.get("vram_limit_breached", False),
        **_sim_columns(metrics),
    }

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel shows non-ASCII notes correctly
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return json_path, csv_path

def _best_columns(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Best configuration of a sweep run (empty cells for a single-configuration run)."""
    b = metrics.get("best_config") or {}
    return {
        "best_config": b.get("label", ""), "best_source_resolution": b.get("source", ""), "best_input_size": b.get("input", ""),
        "best_conf": b.get("conf", ""), "best_rule": b.get("rule", ""), "best_reason": b.get("reason", ""),
        "best_stability": "" if b.get("stability") is None else b.get("stability"), "configs_tested": b.get("n_configs", ""),
    }

def _sim_columns(metrics: Dict[str, Any]) -> Dict[str, Any]:
    sim = metrics.get("simulated") or {}
    return {
        "model_gflops": (metrics.get("model_profile") or {}).get("gflops"),
        "sim_device": sim.get("device", ""),
        "sim_est_fps": sim.get("est_fps", ""),
        "sim_latency_mean_ms": sim.get("latency_ms", {}).get("mean", ""),
        "sim_inference_ms": sim.get("inference_ms", ""),
    }

COMPARISON_CSV_COLUMNS = ["target", "model", "notes", "best_config", "best_source_resolution", "best_input_size", "best_conf",
                          "best_rule", "best_reason", "best_stability", "configs_tested", "frames", "fps", "latency_mean_ms", "latency_p95_ms", "peak_ram_mb",
                          "model_gflops", "sim_device", "sim_est_fps", "sim_latency_mean_ms", "total_detections", "report_id", "error"]
NOTES_COL_WIDTH = 32  # characters of the Notes column in the console comparison table

def save_comparison_report(runs: List[Dict[str, Any]], output_path: str | Path, csv_too: bool = False) -> Path:
    """Writes all runs of one benchmark session (e.g. several models on one target) to a single JSON file
    (and, with csv_too, to a CSV with the same name: one row per run, notes included)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"created": datetime.now().isoformat(timespec="seconds"), "runs": runs}, f, indent=2, ensure_ascii=False)
    if csv_too:
        with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COMPARISON_CSV_COLUMNS)
            writer.writeheader()
            for r in runs:
                sim = r.get("simulated") or {}
                row = {
                    "target": r.get("target", ""), "model": r.get("model", ""), "notes": r.get("notes", ""),
                    **_best_columns(r),
                    "frames": r.get("frames_processed", ""), "fps": r.get("throughput_fps", ""),
                    "latency_mean_ms": (r.get("latency_ms") or {}).get("mean", ""),
                    "latency_p95_ms": (r.get("latency_ms") or {}).get("p95", ""),
                    "peak_ram_mb": (r.get("system_resources") or {}).get("peak_ram_mb", ""),
                    "model_gflops": (r.get("model_profile") or {}).get("gflops", ""),
                    "sim_device": sim.get("device", ""), "sim_est_fps": sim.get("est_fps", ""),
                    "sim_latency_mean_ms": (sim.get("latency_ms") or {}).get("mean", ""),
                    "total_detections": r.get("total_detections", ""), "report_id": r.get("report_id", ""),
                    "error": r.get("error", ""),
                }
                writer.writerow(row)
    return out

def print_comparison_table(runs: List[Dict[str, Any]]) -> None:
    if not runs:
        return
    sim_device = next(((r.get("simulated") or {}).get("device") for r in runs if r.get("simulated")), None)
    header = f"{'Model':<40} {'GFLOPs':>7} {'Host ms':>8} {'Host FPS':>8}"
    if sim_device:
        header += f" {'Est. ms':>8} {'Est. FPS':>8}"
    header += f" {'Peak RAM':>9} {'Dets':>5} {'Best configuration':<30} {'Notes':<{NOTES_COL_WIDTH}}"
    print("\n" + "=" * len(header))
    print(f"  MODEL COMPARISON: {runs[0].get('target', '').upper()}" + (f"  (simulated: {sim_device})" if sim_device else ""))
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in runs:
        if "error" in r:
            print(f"{r.get('model', '?'):<40} ERROR: {r['error']}")
            continue
        gflops = (r.get("model_profile") or {}).get("gflops")
        line = (f"{r.get('model', ''):<40} {gflops if gflops is not None else '-':>7} "
                f"{r['latency_ms']['mean']:>8} {r['throughput_fps']:>8}")
        if sim_device:
            sim = r.get("simulated") or {}
            line += f" {sim.get('latency_ms', {}).get('mean', '-'):>8} {sim.get('est_fps', '-'):>8}"
        line += f" {r['system_resources'].get('peak_ram_mb', 0):>9} {r.get('total_detections', 0):>5}"
        line += f" {(r.get('best_config') or {}).get('short', '-'):<30}"
        note = one_line(r.get("notes", ""), NOTES_COL_WIDTH)
        line += f" {note}" if note else ""
        print(console_safe(line))
    print("=" * len(header) + "\n")

def print_summary_table(metrics: Dict[str, Any]) -> None:
    sep = "=" * 65
    sys_res = metrics.get("system_resources", {})
    ram_limit = f"{sys_res.get('ram_limit_mb')} MB" if sys_res.get('ram_limit_mb') else "Unlimited"
    vram_limit = f"{sys_res.get('vram_limit_mb')} MB" if sys_res.get('vram_limit_mb') else "Unlimited"

    ram_status = " [BREACHED]" if sys_res.get("ram_limit_breached") else " [OK]"
    vram_status = " [BREACHED]" if sys_res.get("vram_limit_breached") else " [OK]"

    print("\n" + sep)
    print(f"  BENCHMARK REPORT: {metrics.get('target', 'N/A').upper()}  ")
    print(sep)
    print(f" Model:               {metrics.get('model')}")
    print(f" Processed Frames:    {metrics.get('frames_processed')}")
    print(f" Throughput:          {metrics.get('throughput_fps')} FPS")
    print(f" Total Latency (P50): {metrics.get('latency_ms', {}).get('p50')} ms")
    print(f" Total Latency (P95): {metrics.get('latency_ms', {}).get('p95')} ms")
    print(f" Total Latency (Mean):{metrics.get('latency_ms', {}).get('mean')} ms")
    print(f" Inference Only (Avg):{metrics.get('inference_only_ms', {}).get('mean')} ms")
    print(f" RAM Limit:           {ram_limit}")
    print(f" Peak RAM Used:       {sys_res.get('peak_ram_mb')} MB{ram_status if sys_res.get('ram_limit_mb') else ''}")
    print(f" VRAM Limit:          {vram_limit}")
    print(f" Peak VRAM Used:      {sys_res.get('peak_vram_mb')} MB{vram_status if sys_res.get('vram_limit_mb') else ''}")
    print(f" Avg CPU Load:        {sys_res.get('avg_cpu_percent')} % of target cores")
    if metrics.get("notes"):
        print(console_safe(f" Notes:               {one_line(metrics['notes'])}"))
    if metrics.get("best_config"):
        b = metrics["best_config"]
        print(console_safe(f" Best configuration:  {b['short']}  [{b['rule']}] ({b['n_configs']} tested)"))
    sim = metrics.get("simulated")
    if sim:
        print(sep)
        print(f"  SIMULATED ON: {sim['device']} [{sim['runtime']}]")
        print(sep)
        gflops = (metrics.get("model_profile") or {}).get("gflops")
        print(f" Model Compute:       {gflops} GFLOPs")
        print(f" Est. Inference:      {sim['inference_ms']} ms")
        print(f" Est. Total Latency:  {sim['latency_ms']['mean']} ms (P95 {sim['latency_ms']['p95']} ms)")
        print(f" Est. Throughput:     {sim['est_fps']} FPS")
        ref = sim.get("reference", {})
        print(f" Calibration Anchor:  {ref.get('model')} = {ref.get('latency_ms')} ms ({ref.get('gflops')} GFLOPs)")
    print(sep + "\n")


def print_sweep_table(report: Dict[str, Any]) -> None:
    """Console table of every configuration of a sweep run (the best one is marked with '*')."""
    sw = report.get("sweep")
    if not sw:
        return
    from src.benchmark import sweep as sweep_mod
    simulated = report.get("device_estimate") is not None
    best_id = sw["best"]["config_id"]
    header = (f"  {'Source':<10} {'Input':<9} {'Conf':>5} {'Est FPS' if simulated else 'Host FPS':>8} {'RT':>3} {'GFLOPs':>7} "
              f"{'Dets/fr':>7} {'AgreeF1':>7} {'Temporal':>8} {'Stability':>9} {'Cov':>3} {'Dup':>4}")
    print("=" * len(header))
    print(f"  ALL CONFIGURATIONS ({len(sw['configs'])})   * = best")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    fmt = lambda v, w, nd=3: f"{'n/a' if v is None else (round(v, nd) if isinstance(v, float) else v):>{w}}"
    for e in sorted(sw["configs"], key=lambda e: (-e["source"]["height"], -(e["input"]["h"] * e["input"]["w"]), e["conf"])):
        ag, r = e.get("agreement"), e.get("ratings") or {}
        mark = "*" if e["id"] == best_id else " "
        print(f"{mark} {e['source']['label']:<10} {e['input']['label']:<9} {sweep_mod.conf_text(e['conf']):>5} "
              f"{sweep_mod.entry_fps(e):>8.1f} {'yes' if e['realtime']['realtime_capable'] else 'no':>3} {fmt(e.get('gflops'), 7, 2)} "
              f"{fmt(e['detections']['mean_per_frame'], 7, 2)} {fmt(ag['f1'] if ag else None, 7)} {fmt(e.get('temporal'), 8)} "
              f"{fmt(e.get('stability'), 9)} {fmt(r.get('coverage'), 3)} {fmt(r.get('duplicates'), 4)}")
    print("=" * len(header))
    print(console_safe(f"  Best: {sw['best']['label']}  [{sweep_mod.RULE_NAMES.get(sw['best']['rule'], sw['best']['rule'])}]"))
    print(console_safe(f"  Why:  {sw['best']['reason']}"))
    print()
