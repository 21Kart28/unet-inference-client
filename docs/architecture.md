# Architecture

The client is a PySide6 desktop app that talks to a MONAI Label server over HTTP, keeps its
own results folder on disk, and hands finished segmentations to 3D Slicer. Everything slow
runs on a `QThread` so the window stays responsive.

```mermaid
flowchart LR
    subgraph Client["U-Net Inference Client (PySide6)"]
        UI["MainWindow + tabs<br/>app.py, *_tab.py"]
        Backend["MonaiLabelBackend<br/>backend.py"]
        Metrics["metrics.py<br/>Dice, volume, HD95"]
        History["history.py<br/>manifests + CSV"]
        Report["report.py<br/>Markdown / HTML"]
        subgraph Workers["QThread workers"]
            BatchW["BatchWorker<br/>batch.py"]
            CompareW["CompareWorker<br/>compare_tab.py"]
            StatusW["StatusWorker<br/>app.py"]
            CheckW["CheckWorker<br/>setup_tab.py"]
            ImportW["ImportWorker<br/>dicom_import.py"]
        end
    end

    Server["MONAI Label server<br/>/info/ /datastore/ /infer/"]
    Studies[("Studies folder<br/>images + labels/final")]
    Results[("Results folder<br/>pred.nrrd, manifest.json,<br/>metrics.json, history CSV")]
    Slicer["3D Slicer<br/>--python-script"]
    Proc["server.py<br/>ServerProcess"]

    UI -- signals --> Workers
    BatchW --> Backend
    StatusW --> Backend
    CheckW --> Backend
    Backend -- HTTP --> Server
    Server --- Studies
    Backend -- writes --> Results
    CompareW --> Metrics
    Metrics -- reads --> Results
    Metrics -- reads --> Studies
    ImportW -- NRRD --> Studies
    UI --> History
    History -- scans --> Results
    UI --> Report
    UI -- subprocess --> Slicer
    Slicer -- loads --> Results
    UI -- subprocess --> Proc
    Proc -- starts --> Server
```

## Pieces

| Piece | Module | Thread | Role |
| --- | --- | --- | --- |
| Main window and tabs | `app.py`, `setup_tab.py`, `compare_tab.py`, `model_compare_tab.py` | GUI | Wires signals; no inference or file scanning of its own. |
| Backend | `backend.py` | caller's | `Backend` interface, a `FakeBackend` for tests, and `MonaiLabelBackend`, which calls `/info/`, `/datastore/` and `/infer/` with `requests`. |
| Batch worker | `batch.py` | QThread | Runs one case after another, emits progress, supports cancel and resume. |
| Compare worker | `compare_tab.py` | QThread | Dice, volume and HD95 per result; per-class rows for multi-label maps. |
| Metrics | `metrics.py` | worker | SimpleITK + NumPy; crops to the union bounding box before HD95 so full scans stay fast. |
| History | `history.py` | GUI | Reads `manifest.json` from each result folder and appends to `inference_history.csv`. |
| Model comparison | `model_compare.py` | GUI | Pairs the newest Dice per case for two models and summarizes the delta. |
| Reports | `report.py` | GUI | Builds one `Report` structure and renders it as Markdown or HTML. |
| DICOM import | `dicom_import.py` | QThread | Finds series with SimpleITK's `ImageSeriesReader` and writes an NRRD into the studies folder. |
| Server process | `server.py` | GUI + reader QThread | Starts and stops a local MONAI Label server and streams its output. |
| Slicer launcher | `slicer_view.py` | subprocess | Writes a small Python script and starts Slicer with `--python-script`. |
| Setup checks | `setup_check.py`, `setup_tab.py` | QThread | Each install check returns OK, optional or problem with a fix hint. |
| Settings | `config.py` | GUI | Dataclass persisted as JSON under the user's config folder. |
| Logging | `logs.py` | any | Rotating file under the config folder; the log pane mirrors into it. |

## Threading rules

- A worker is a `QObject` with signals, moved onto a `QThread`; the thread's `started` signal
  calls `run()`. Results come back through signals, never shared state.
- Workers catch their own exceptions and emit a plain sentence; the GUI never sees a traceback.
- `closeEvent` quits and waits on every thread so the app exits cleanly.

## Data on disk

```
<results>/
  inference_history.csv                one row per attempt
  <case>_<model>_<res>_ep<N>_dice<D>/  one folder per result
    pred.nrrd                          prediction from the server
    manifest.json                      case, model, checkpoint, server, time
    metrics.json                       Dice, volumes, HD95, per-class rows
<studies>/
  <case>.nrrd                        images served by MONAI Label (DICOM imports land here)
  labels/final/<case>.nrrd           ground truth used by the Compare tab
```
