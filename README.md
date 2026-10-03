# Model Inference Benchmark: Edge Hardware Inference & Benchmark Harness

A modular framework designed to run and benchmark crime detection models (YOLO, ONNX, TensorRT, RKNN, HailoRT) on simulated edge hardware environments via Docker. Target platforms and execution constraints are configured via `targets.yaml`.

---

## 📁 Directory Structure

```text
model-inference-benchmark/
├── targets.yaml             # Target hardware specifications & benchmark policies
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── configs/
│   └── detection.yaml       # Detection parameters (conf_threshold, iou, input size)
│
├── models/                  # Storage for model weights
│   ├── README.md            # Guidelines for weights formats (.pt, .onnx, .engine, .rknn, .hef)
│   └── best-yolo11-seg.pt   # Preloaded crime detection / segmentation model
│
├── videos/                  # Video data directory
│   ├── README.md            # Video input/output guidelines
│   ├── input/               # Put raw CCTV footage / sample videos here
│   └── output/              # Annotated inference videos saved here
│
├── docker/                  # Docker containerization & simulation
│   ├── Dockerfile.x86_cpu   # x86-64 CPU container
│   ├── Dockerfile.arm64_cpu # ARM64 multi-architecture container (QEMU emulation)
│   ├── Dockerfile.jetson    # NVIDIA Jetson JetPack / TensorRT container
│   ├── docker-compose.yml   # Multi-service target runner
│   └── .dockerignore
│
├── src/                     # Core application source code
│   ├── config.py            # targets.yaml parser & emulation validator
│   ├── runtimes/            # Hardware-specific execution adapters
│   │   ├── base.py          # Abstract BaseDetector interface
│   │   ├── pytorch_runner.py# PyTorch / Ultralytics runner
│   │   ├── onnx_runner.py   # ONNX Runtime CPU/CUDA engine
│   │   ├── jetson_runner.py # Jetson TensorRT adapter
│   │   ├── rknn_runner.py   # Rockchip RK3588 NPU adapter
│   │   └── hailo_runner.py  # Hailo-10H AI module adapter
│   ├── video/               # Video frame extraction & bounding box annotation
│   │   ├── reader.py        # Video stream reader
│   │   └── annotator.py     # Detections & telemetry overlay
│   ├── simulation/          # Edge-device simulation
│   │   ├── model_profile.py # Model GFLOPs / params counter (.onnx, .pt)
│   │   ├── device.py        # Per-target latency model (calibrated from targets.yaml)
│   │   ├── simulated_detector.py # Runs the real model, attaches est. device timings
│   │   └── estimates.py     # Model x target estimate table
│   ├── benchmark/           # Profiling & metric reporting
│   │   ├── profiler.py      # Per-frame records, latency percentiles, FPS, RAM/CPU, detection stats
│   │   ├── report.py        # Legacy results/metrics JSON/CSV exporter & console tables
│   │   ├── report_builder.py# Builds the report.json (schema 1.0): environment, verdicts, real-time check
│   │   ├── export.py        # Report bundle writer (json, csv, md, sample frames, html)
│   │   ├── html_report.py   # Self-contained HTML report (inline CSS + SVG charts)
│   │   ├── suite.py         # Benchmark suite runner: manifest (suite.json), stop/resume, CSV/JSON exports
│   │   ├── suite_site.py    # Multipage benchmark report site (overview, devices, models, methodology, print)
│   │   ├── sweep.py         # Sweep settings, quality proxies (agreement / temporal / stability), best-configuration rule
│   │   ├── sweep_runner.py  # Executes the sweep passes (model loaded once) and builds the per-configuration entries
│   │   ├── sweep_html.py    # Best-configuration banner, configuration table, charts, sample grid
│   │   ├── ratings.py       # Manual ratings store (results/ratings/ratings.json)
│   │   └── rerender.py      # Re-renders reports / suites after ratings changed (no model is run again)
│   └── utils/
│       └── logger.py        # Standardized logging
│
├── scripts/                 # CLI entry points
│   ├── run_inference.py     # Run detection on video files or live streams
│   ├── run_benchmark.py     # Run formal benchmark (one or several models) on a target + write report
│   ├── run_benchmark_matrix.py # Benchmark suite: every model x every device -> multipage report (also --resume / --render-only)
│   ├── rate_configs.py      # Manual quality ratings of benchmark configurations (+ re-render of the reports)
│   ├── check_sweep_equivalence.py # Proves confidence filtering == re-running; self-test of the quality functions
│   └── estimate_models.py   # Instant model x target latency estimates (no inference)
│
└── results/                 # Output benchmark metrics & logs
    ├── ratings/             # ratings.json: your manual detection-quality ratings per video + model + configuration
    ├── metrics/             # Exported JSON and CSV benchmark logs
    ├── reports/             # One folder per benchmark report (+ suite_<timestamp>/ benchmark suites)
    └── logs/
```

---

## 🎯 Target Hardware Selection (`targets.yaml`)

You can manually choose and test any target platform at any time:

| Target ID | Simulated device | Runtime modelled | CPU cores | Calibration anchor |
|---|---|---|---|---|
| `x86-cpu` | Host (measured, not simulated) | ONNX Runtime / PyTorch | all | — |
| `arm64-cpu` | Raspberry Pi 5 | ONNX Runtime CPU FP32 | 4 | YOLO26n = 126 ms |
| `jetson` | Jetson Orin Nano 8GB (Super) | TensorRT FP16 | 6 | YOLO26n = 4.57 ms |
| `rk3588-npu` | RK3588 (Rock 5B) NPU | RKNN INT8 | 4 | YOLO26n = 41.2 ms |
| `hailo10h` | Hailo-10H on a Raspberry Pi 5 | HailoRT INT8 | 4 | YOLOv8n = 2.67 ms |

To list all available targets from the CLI:
```bash
python scripts/run_benchmark.py --list-targets
```

---

## 🧪 How Edge Hardware Is Simulated

The target boards (and their NPUs) are not available on the dev machine, so each target is simulated in two layers:

1. **Functional (real):** the `.onnx` / `.pt` model really runs on the host, inside the target's Docker service, which is limited to the device's CPU cores (`cpus`) and RAM (`mem_limit`), with the runtime's thread count set to `hardware.cpu_cores`. Detections, RAM usage and CPU-side behaviour are measured for real.
2. **Timing (estimated):** the on-device latency is derived from a published benchmark of a YOLO-class model on that exact device (`hardware.reference` in `targets.yaml`):
   - `est. inference = max(min_inference_ms, anchor_ms × model_GFLOPs / anchor_GFLOPs)`
   - `est. pre/post = host pre/post ms × host CPU score / device CPU score`

The live player shows the estimated device FPS/latency (and the host's actual latency), and holds frames so playback runs at the device's speed whenever the host is faster than the device. Benchmarks report host and estimated-device numbers side by side.

**Accuracy:** roughly ±30–50% vs. real hardware. Use it to rank models and to see which cannot reach real time on a device, not to sign off on deployment. Not modelled: INT8 quantization accuracy loss, NPU operator fallbacks to CPU, thermal throttling, and interface bottlenecks (e.g. Hailo on the Pi 5's PCIe x1).
Device-native formats (`.engine`, `.rknn`, `.hef`) still use the vendor runtimes and only work on real hardware.

```bash
# Instant estimate of every model on every target (no inference)
python scripts/estimate_models.py

# Benchmark several models on a simulated target and compare
python scripts/run_benchmark.py --docker --target rk3588-npu --frames 50   --model models/HumanDetection_light_input_640.onnx models/HumanDetection_server_input_640.onnx
```

To recalibrate, edit `hardware.reference` (for example with your own on-device measurement) and `simulation.host.cpu_single_core_score` when running on another machine. Keep `cpu_cores` / `ram_limit_mb` in sync with `docker/docker-compose.yml`.

---

## 📊 Benchmark Reports (export)

Every benchmark run produces an exportable report bundle in `results/reports/<report_id>/`, where
`report_id = <target>__<model>__<YYYYmmdd_HHMMSS>`.

**In the web app:** pick a target and a model (in *Benchmark* mode you can Ctrl/Shift-click several models: one report each),
set *Frames* and the video, then press **📊 Benchmark & Export Report**. A progress bar follows the run; when a model
finishes, a report card appears with the verdict, the key numbers and the export buttons
(**Download ZIP**, **Open HTML report**, JSON, CSV, Summary). The **Past Reports** table below lists every earlier report
(filterable by target and model) with the same links. Benchmarks run in Docker (hardware simulated) or on the host Python
(environment toggle), exactly like the CLI.

**Notes per model/device pair.** Under the button, **📝 Notes for this benchmark (optional)** has one text box per selected model on the selected device
(several models in Benchmark mode: one box each). Whatever you type is stored in the report as a **Notes** field and shown in the HTML report (box under the
title), `report.json` (`"notes"`), `summary.md` (`- Notes:` line), the `results/metrics/benchmark_*.csv` and comparison files (`notes` column), the report card,
the comparison table and the **Past Reports** table (Notes column). An empty note leaves the cell empty; the column is always there. Notes are plain text
(any language, line breaks kept, at most 2000 characters, longer text is cut and the counter says so) and are always HTML-escaped. They are remembered per
model + device in the browser (`localStorage`), so a re-run is prefilled; **clear** empties one.

**What is in a bundle**

| File | Content |
|---|---|
| `report.html` | Self-contained report (no CDN or external files; light/dark theme, print to PDF works): Notes box, verdict badge, KPI tiles, charts (latency per frame host vs. estimated device, latency histogram, stage breakdown, detections per frame and per class), detail tables, sample frames, method and accuracy disclaimer |
| `report.json` | Full machine-readable report (schema 1.0, below) |
| `frames.csv` | One row per frame: decode / pre / inference / post ms, host and estimated device latency, detections, per-class counts, max confidence, RSS MB, CPU % |
| `summary.md` | Short paste-ready summary of the headline numbers and the verdict |
| `samples/frame_XXXXXX.jpg` | Up to 3 annotated frames (the ones with the most detections) |

The web app builds the ZIP on the fly from that folder (`GET /api/reports/<id>/download`); nothing else is stored.

**What is measured, what is estimated**

- Measured on the host (inside the target's Docker limits): latency percentiles per stage, throughput, wall-clock FPS, detections and
  confidences, model load time, first-inference time, RSS before/after load and peak RAM, CPU load relative to the target cores.
  Video decoding is timed separately and excluded from latency/FPS. Frames are streamed, so RAM reflects the model and not a preloaded video.
  `throughput_fps` counts processing time only; `wall_fps` includes decoding and bookkeeping.
- Estimated (simulated targets only): on-device inference and total latency and FPS, from the calibration anchor in `targets.yaml`.
  The report always states the ±30-50% accuracy and what is not modelled. `x86-cpu` is host-measured only.
- Real-time verdict: required FPS is `benchmark.required_fps` in `configs/detection.yaml` if set, else the video's own FPS, else 25.
  factor = (estimated device FPS, or host FPS when not simulated) / required FPS. `>= 1` is real-time; `0.5 - 1` warns; below `0.5` fails.
  The report also says "analyses 1 of every N frames" when it cannot keep up.
- Models whose output head is not YOLOv8/11 or end-to-end (e.g. face, pose, embedding models) are timed but produce no detections;
  the verdict then says **output format not decoded** instead of implying the model found nothing.

**`report.json` overview (`schema_version` 1.0):** `report_id`, `created`, `notes` (user text for this model/device pair, `""` when none), `environment` (docker/local, host, CPU, cgroup limits, package versions),
`target` (device, runtime, cores, RAM/VRAM budget, calibration anchor, method note and disclaimer), `model` (file, size, sha256 prefix, GFLOPs, params,
input size, classes, output format), `config` (thresholds, frames, warmup, source), `cold_start`, `host_performance`, `device_estimate` (null when not simulated),
`realtime`, `resources`, `detections`, `verdict` (`overall` + list of `{level, message}`), `frames_csv`, `samples`.

### Sweeps, best configuration and manual ratings

Every exportable benchmark (**📊 Benchmark & Export Report** for one or several models, and the **🧮 Benchmark All Scenarios** suite) tests each
model/device pair over a **sweep** and reports the **best configuration** per pair. A configuration is

| Dimension | Values (defaults in `configs/detection.yaml` -> `benchmark.sweep`) | Notes |
|---|---|---|
| Source (camera) resolution | frame height 720, 1080, native (480 also available) | Every decoded frame is downscaled to this height (aspect ratio kept) **outside** the timed section. Values above the video's own height are skipped. |
| Model input size | 480, 640, 800 (+ the model's own size); 320 / 960 in the *Full* preset | **Only for models with a dynamic input**: Ultralytics `.pt` weights and ONNX graphs with symbolic H/W (detected at run time, e.g. `best-yolo11-seg.pt`, `FaceDetection_input_dynamic.onnx`). Models with a static input (Gun, Helmet, HumanDetection, face-pose, mask, ...) are *locked*: ORT rejects other sizes, so they run at their own size and the report says so. GFLOPs and the device latency estimate are recomputed for every input size. |
| Confidence | 0.25, 0.35, 0.5 (the run's / UI confidence is always added) | **No extra inference**: the model runs once per (resolution, input size) at the lowest threshold and higher thresholds drop boxes below them. Greedy NMS is consistent under score filtering, so this gives exactly the boxes of a run at that threshold (`python scripts/check_sweep_equivalence.py` shows it for the ONNX YOLO head, the end-to-end head and the `.pt` model). Postprocess time is measured at the lowest threshold and reused (slight over-estimate for the higher ones). |

The model is loaded **once** per process; the input size is switched in place, the frames of a source resolution are decoded once and reused when they fit in
400 MB (the cache is subtracted from the peak-RAM measurement). Cost is `resolutions x input sizes` passes of `frames` frames; confidence values cost nothing.

**Per-configuration metrics** (beside the usual latency / FPS / RAM / detection numbers): *agreement F1* against the reference configuration (highest
resolution x largest input size x the run's confidence; boxes of the same class matched at IoU >= 0.5; the reference scores 1.0 by definition, so agreement
peaks near the run's confidence), *temporal consistency* (share of boxes that reappear, same class, IoU >= 0.3, in the next frame) and
*stability = 0.5 x agreement F1 + 0.5 x temporal consistency*. For models whose output is not decoded these are n/a.

**Best configuration rule** (also written on every report and on the suite's methodology page):
1. If you entered manual ratings for this video and model, only the **rated** configurations compete: highest *coverage*, then fewest *duplicates* (blank
   ranks after entered), then real-time on this device, then stability, then the higher source resolution. The automatic pick is shown for comparison when it differs.
2. Otherwise the **real-time** configurations (estimated device FPS, host FPS for the host-measured target, at least the required FPS) compete: highest
   stability (equal to two decimals = tie), then higher source resolution, then larger input size.
3. If nothing is real-time the fastest one is shown and flagged *not real-time at any tested setting*.

**Manual ratings.** After a run press **⭐ Rate detection quality** on a report card, in the **Past Reports** table or on a suite card. The view lists every
configuration of each model with the **same sample frames** (the ones with the most detections in the reference configuration, annotated per configuration; click to
enlarge) and two inputs per configuration: **Coverage 1-5** (5 = every person is detected, including people far back or partly covered; 1 = many missed) and
**Duplicates** (number of duplicate boxes you saw; lower is better). Blank = not rated. *Save* stores them in `results/ratings/ratings.json` (key
`<video>|<model>|<source res>|<input HxW>|<conf>`) and re-renders every report and suite of that video and model: no model is run again, and the best
configuration switches to your rating (clearing the ratings switches back to the automatic pick). Detections do not depend on the simulated device, so a rating
applies on every device. A running suite picks the ratings up when it finishes.

**What the reports show.** `report.html` of a sweep run: a *best configuration* banner (configuration, rule, reason, the automatic pick when different, ratings used),
a table of **all configurations** (FPS, real-time, P95, GFLOPs, RAM, detections, agreement, temporal, stability, ratings; best row highlighted), charts (FPS vs source
resolution with one line per input size, latency breakdown per configuration, detections and stability vs confidence), sample frames of every configuration, the selection
rule and the usual detail sections for the best configuration. `report.json` keeps the top-level sections for the **best** configuration (so existing consumers keep working)
and adds `sweep` (`dimensions`, `reference_config`, `configs[]` with every metric / rating / sample image, `best` with `rule`, `reason`, `auto_pick`); `frames.csv` has one row per
(configuration, frame) with `config_id`, `source`, `input`, `conf` columns; `summary.md` has the best configuration and a compact table; the `results/metrics` JSON / CSV and the
comparison files have `best_config*` columns. Suite pages: the overview heatmap, real-time and RAM matrices show the best configuration under every value (a star = chosen by
your rating; hover for the reason); device pages list "Best configuration" and "Why" per model and link to the run report that holds all configurations; model pages add a per-configuration
table (quality plus the estimated FPS on every device); `results.csv` / `results.json` have one row per pair **and configuration** with `is_best`, `selection_rule`, `selection_reason`, ratings and notes.
Reports and suites made before sweeps (no `sweep` data) still open and re-render as a single configuration.

**Web app.** Both the benchmark area and the suite panel have a *Sweep* box: checkboxes for the source resolutions and input sizes, an editable list of confidence thresholds, the
presets *Quick* (native + 720p, model default size, 0.25 / 0.35 / 0.5), *Full* and *Defaults*, **No sweep** (one configuration, the old behaviour), and a live count of configurations with a rough ETA.
The live progress and the suite grid show the current pass (for example `720p · 640 · pass 2/3`).

**CLI**

```bash
# default sweep (configs/detection.yaml), best configuration + a table of all configurations on the console
docker compose -f docker/docker-compose.yml run --rm -T -e YOLO_CONFIG_DIR=/tmp rk3588-npu python scripts/run_benchmark.py \
  --target rk3588-npu --model models/best-yolo11-seg.pt --video videos/input/sample.mp4 --frames 50
#   --source-heights 720 1080 native   --input-sizes 480 640 800 (or "default")   --conf-thresholds 0.25 0.35 0.5   --sweep-preset quick|full   --no-sweep
python scripts/run_benchmark_matrix.py --targets rk3588-npu jetson --models models/a.onnx models/b.pt --sweep-preset quick --frames 10
python scripts/run_benchmark_matrix.py --render-only suite_20261003_120000      # re-renders the suite AND its per-run reports (new ratings everywhere)

python scripts/rate_configs.py show --suite suite_20261003_120000                # configurations, FPS, stability and ratings per model
python scripts/rate_configs.py set --video v.mp4 --model m.pt --source 720p --input 640x640 --conf 0.35 --coverage 5 --duplicates 0
python scripts/rate_configs.py import my_ratings.json                            # {"ratings": [{video, model, source, input, conf, coverage, duplicates}, ...]}
python scripts/rate_configs.py clear --model m.pt                                # back to the automatic pick
python scripts/rate_configs.py rerender --all                                    # rebuild every sweep report / suite from the stored ratings
python scripts/check_sweep_equivalence.py --self-test                            # hand-made cases of the quality functions and the selection rule
```

**CLI (single run flags)**

```bash
# One model on one simulated device (report goes to results/reports/<id>/)
docker compose -f docker/docker-compose.yml run --rm -T rk3588-npu python scripts/run_benchmark.py \
  --target rk3588-npu --model models/HumanDetection_light_input_640.onnx \
  --video videos/input/sample.mp4 --frames 100 --warmup 5 --conf 0.35
# or from the host: python scripts/run_benchmark.py --docker --target rk3588-npu ...
```

`run_benchmark.py` flags: `--conf` (confidence override), `--no-report` (skip the bundle; the legacy `results/metrics` files are always written),
`--summary-json` (all runs of the invocation, incl. `report_id` and failures), `--notes "text"` (note for the single `--model` on `--target`) and
`--notes-file <path>` (UTF-8 JSON `{"<model file>": {"<target>": "note"}}`, for several models/targets; a plain string instead of the target object applies to
every target; relative paths are relative to the project root and, with `--docker`, must be inside it). Free text with quotes, line breaks or non-ASCII characters is
safest in a notes file: the web app always uses one (`results/reports/_notes/notes_<timestamp>.json`, deleted when the job ends). With `--docker`, `--notes` text is
handed to the container through such a temporary file as well. Example: `--notes-file results/my_notes.json` with
`{"Gun_Detection_input_640.onnx": {"rk3588-npu": "light model for the entrance camera"}}`. A multi-model run without a comparison file also writes
`results/metrics/comparison_<target>_<time>.csv` (one row per model, `notes` included; CSVs with notes are UTF-8 with BOM so Excel shows non-ASCII text). While running it prints `PROGRESS <done>/<total>` and `REPORT <id>` lines (used by the web app).

**Benchmark all scenarios: every model on every device (multipage report)**

One run benchmarks every selected model on every selected device from `targets.yaml` (default: all runnable models x all targets) and
writes a navigable, self-contained report site. It works from the web app (**🧮 Benchmark All Scenarios**) and from the CLI.

*Web app.* Press **🧮 Benchmark All Scenarios** under "Benchmark & Export Report". A setup panel lists the models (broken or device-native ones are flagged and
unchecked), the devices, frames per run (default 30, **Quick** = 10), warmup, video and confidence, plus a rough ETA line. **Start suite** runs the devices one
after another (one Docker container per device, models sequential inside it: parallel runs would corrupt the timings). While it runs you get an overall progress
bar (runs done, elapsed, ETA from the measured run times), the current device / model / frame, and a live model x device grid whose cells change colour as they
finish. **Stop** works at any time: the container is removed, the remaining runs are marked *cancelled* and the partial report is still built. When it is finished
or stopped a card shows the counts and the fastest pair with **Open report**, **Print all (PDF)**, **Download ZIP**, **CSV**, **JSON** and, for an incomplete suite,
**Resume** (reruns only the runs that are not done). Earlier suites are listed under **Benchmark Suites**. The Docker / Host Python toggle on the left applies.
Do not start a live session while a suite runs (the app refuses; timings would be disturbed).

*Notes.* The setup panel has **📝 Notes per model/device pair (optional)**: collapsed by default with an "N of M filled" counter, a filter box, rows grouped by
device, one per checked model x checked device (it follows the checklists, typed text is kept). Each note appears as a **Notes** column on the device pages
(ranking table) and model pages (per-device table; the former "Notes" column with the automatic verdict messages is now called **Checks**), in the
failed / skipped list and in the "Best model per device" table of the overview, as a small &#9998; marker on the heatmap cells of that pair (the note is in the
cell's tooltip), in `print.html`, in `results.csv` / `results.json` (`notes` column/field for every cell) and in the per-run report of that pair. Empty notes
leave empty cells. They are stored in `suite.json` (every cell has `"notes"`, plus the original input in `config.notes`), so **Stop / Resume** keep them.
To change a note afterwards, edit that cell's `"notes"` in `suite.json` and run `python scripts/run_benchmark_matrix.py --render-only <suite_id>`: the site, CSV and JSON
are regenerated (the per-run `runs/<id>/report.*` keep the note they were created with). Suites created before this feature simply show empty Notes cells.

*CLI.*

```bash
python scripts/run_benchmark_matrix.py --targets all --models all --frames 30 --video videos/input/sample.mp4
python scripts/run_benchmark_matrix.py --targets arm64-cpu jetson --models models/a.onnx models/b.onnx --frames 20 --local
python scripts/run_benchmark_matrix.py --resume suite_20260930_120000       # rerun only cells that are not ok / skipped
python scripts/run_benchmark_matrix.py --render-only suite_20260930_120000  # rebuild the site from suite.json
python scripts/run_benchmark_matrix.py --targets rk3588-npu jetson --models models/a.onnx models/b.onnx --notes-file results/my_notes.json
```

Ctrl+C / SIGTERM stops the suite cleanly (container removed, rest cancelled, report rendered). A model that cannot load (for example the broken
`HumanDetection_input_640.onnx`, whose `model.onnx.data` is missing) or a crashed container gives a *failed* cell with the reason and never aborts the suite.
`.engine/.rknn/.hef` models get *skipped* cells (they need the vendor runtime on real hardware).

*Folder layout* (`results/reports/suite_<YYYYmmdd_HHMMSS>/`; the whole folder is what the ZIP contains):

| Path | Content |
|---|---|
| `index.html` | Overview: KPIs, estimated-FPS heatmap (model x device, colour = real-time / >= 5 FPS / slower, every cell links to its run), best model per device, real-time matrix, peak-RAM-vs-budget heatmap, failed / skipped list |
| `devices/<target>.html` | One page per device: specs and calibration anchor, ranking of all models (FPS, P50/P95, CPU-side vs inference ms, RAM vs budget, detections, verdict), charts (FPS, latency breakdown, RAM with budget line) |
| `models/<model>.html` | One page per model: model info, results on every device, FPS / RAM / detections per device, cold start, verdicts |
| `method.html` | How the simulation works, calibration anchors with sources, container limits, colour and verdict rules, what is not modelled |
| `print.html` | All of the above in one document, page break per page, for the browser's Print -> Save as PDF |
| `runs/<report_id>/` | The normal per-run bundle (report.html with a "back to suite" link, report.json, frames.csv, summary.md, one sample frame linked as `samples/*.jpg`) |
| `suite.json` | Manifest: config, per-cell status, key metrics and `notes`, timings; rewritten atomically after every run, so partial suites always render |
| `results.csv`, `results.json` | One row per cell (model x device) with all key metrics and the `notes` of the pair (the CSV is UTF-8 with BOM); replaces the old `matrix.csv` / `matrix.json` |
| `raw/<target>.json` | Raw run summaries of the harness |

All pages use inline CSS and SVG, relative links and no scripts or CDNs: the folder can be zipped, e-mailed or opened from disk. `run_benchmark.py` gained
`--report-root`, `--max-samples`, `--link-samples`, `--back-link` and `--no-metrics-files` for this (defaults unchanged for single runs).

*Runtime.* Every run costs container start (once per device), process start and model load, and `(warmup + frames)` host inferences that are slowed by the
device's CPU/RAM limits (the small Raspberry Pi 5 and Hailo budgets are the slowest). The ETA shown is a rough guess: on the reference laptop (i7-8550U, Docker Desktop) 14 models x 5 devices with 3 frames + 1 warmup took about 9.5 minutes and produced an 11.7 MB ZIP; with the default 30 frames per run extrapolating the measured per-frame times gives roughly 18 minutes for the same matrix (large models such as `HumanDetection_server` at 75 GFLOPs dominate). Use **Quick (10 frames)** or fewer models/devices for a fast first look.
The numbers are estimates with about +/-30-50% error versus real hardware (see the methodology page): use them to rank models and spot the ones that cannot reach real time.

---

## 🖥️ Interactive Web Dashboard & Live Player

A lightweight web app and video player is provided via [app.py](app.py):

### Key Features:
- **Live Video Streaming (MJPEG)**: Displays bounding boxes, detections count, and latency rendered live at full FPS.
- **Playback & Seek Controls**:
  - Jump Forward/Backward: `[ ⏪ -10s ]`, `[ ⏪ -2s ]`, `[ +2s ⏩ ]`, `[ +10s ⏩ ]`.
  - Interactive Scrubbing / Timeline slider.
  - Play / Pause toggle.
- **Save Output Toggle**: A dedicated checkbox allows you to choose whether to write the annotated video (`.mp4`) to disk or run purely in-memory.
- **Hardware Target & Model Selector**: Choose from `x86-cpu`, `arm64-cpu`, `jetson`, `rk3588-npu`, etc.
- **RAM & VRAM Limits**: Enforce custom hardware memory budgets.
- **Benchmark & Export Report**: one click benchmarks the selected model(s) on the selected target and produces a downloadable report ZIP; past reports stay listed (see [Benchmark Reports](#-benchmark-reports-export)).
- **🧮 Benchmark All Scenarios**: every model on every device with live progress, Stop / Resume and a multipage report (overview, one page per device and model, methodology, printable all-pages document, CSV/JSON) downloadable as one ZIP.

To launch the dashboard:
```bash
python app.py
```
Opens **`http://localhost:5000`** in your browser.

---

## 🚀 Quickstart (CLI)

### 1. Local Setup
```bash
pip install -r requirements.txt
```

### 2. Run Inference on a Video
Place your video into `videos/input/sample.mp4`, then run:
```bash
python scripts/run_inference.py \
  --target x86-cpu \
  --model models/best-yolo11-seg.pt \
  --video videos/input/sample.mp4 \
  --output videos/output/annotated_sample.mp4
```

### 3. Run Benchmark with RAM / VRAM Limits
Execute performance profiling while enforcing simulated hardware memory budgets:
```bash
# Run on Host:
python scripts/run_benchmark.py \
  --target jetson \
  --model models/best-yolo11-seg.pt \
  --frames 100 \
  --max-ram-mb 4096 \
  --max-vram-mb 2048

# Run inside Docker Container:
python scripts/run_benchmark.py \
  --target arm64-cpu \
  --docker \
  --frames 100 \
  --max-ram-mb 2048
```
- **`--docker`**: Automatically routes the job into the hardware's Docker container via Docker Compose.
- **`--max-ram-mb`**: Cap and track RAM usage (e.g. `2048` for 2GB edge boards).
- **`--max-vram-mb`**: Hard-cap PyTorch GPU VRAM memory fraction and monitor GPU allocation.

---

## 🐳 Docker Execution

### Direct Docker Compose Usage
```bash
cd docker

# Run x86 CPU benchmark container
docker compose run --rm x86-cpu

# Smoke-test ARM64 container under QEMU emulation
docker compose run --rm arm64-cpu
```

### Build Individual Images
```bash
# x86-64 CPU image
docker build -t model-inference-benchmark:x86-cpu -f docker/Dockerfile.x86_cpu .

# Multiarch ARM64 image (via QEMU)
docker buildx build --platform linux/arm64 -t model-inference-benchmark:arm64-cpu -f docker/Dockerfile.arm64_cpu .
```
