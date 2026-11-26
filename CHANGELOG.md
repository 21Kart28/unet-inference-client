# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Batch Resume button that continues a cancelled batch from the first case it never reached.
- Models tab: Dice per case for two models side by side, with the delta, a mean and a CSV export.
- Per-class Dice and volume for multi-class label maps, shown as a table in the Compare tab
  and saved into each result's metrics.json.
- `--models` flag on the mock server for trying the Models tab without MONAI.
- ruff, mypy and pre-commit configuration; CI runs lint and type checks before the tests.
- Coverage measured in CI with a 75% floor and a Codecov badge.
- Dockerfile and compose service for the mock server.
- Export report buttons in the Compare and Models tabs: a Markdown or HTML report with server,
  model, per-case table, per-class table when present and summary Dice statistics.

### Changed

- Result folders are named after the model as well as the checkpoint, so two models with the
  same epoch and Dice no longer share a folder.
- "Skip cases that already have a result" now checks the results folder for any stored result
  from the current model and server, rather than only the current checkpoint's folder.
- The batch worker, planning and summary live in `batch.py`; `app.py` only wires signals.

### Fixed

- The coverage database is no longer tracked in git.

## [0.1.0] - 2026-11-23

### Added

- PySide6 desktop client for a MONAI Label U-Net segmentation server: Setup, Run, Batch,
  Results, Compare, Server and Settings tabs.
- Single and batch inference on a worker thread, with per-case and total timers and cancel.
- Dice, volume, volume difference and HD95 metrics against ground truth, with CSV export.
- Result manifests recording model, checkpoint epoch, validation Dice, weights hash and time.
- 3D Slicer viewing of any prediction over its source scan, downloading the scan on demand
  from a remote server.
- Local MONAI Label server start and stop from the app, on Windows and Linux.
- Setup tab that checks the install and says how to fix each problem.
- Mock MONAI Label server for tests and for trying the app without MONAI.
- GitHub Actions running pytest on Windows and Ubuntu.

[Unreleased]: https://github.com/21Kart28/unet-inference-client/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/21Kart28/unet-inference-client/releases/tag/v0.1.0
