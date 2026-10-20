# U-Net Inference Client

A desktop tool for running 3D deep learning inference on volumetric medical scans.

Built during an ML research internship at Baylor College of Medicine / Texas Children's
Hospital. Running inference there meant activating a virtual environment, starting a
server from the terminal, and chaining shell scripts by hand for every scan. This replaces
that with a GUI: start the server, pick a scan, run the network on GPU, batch across a
dataset, and review predictions in 3D Slicer.

**Stack:** Python, PySide6/Qt, PyTorch, MONAI Label, REST, CUDA, 3D Slicer

## What it does

- **Server control.** Start and stop a MONAI Label server from the app, and switch which
  dataset it serves without touching a terminal.
- **Single inference.** Pick a case, run the model, watch a live elapsed timer. The UI
  stays responsive because the job runs on a worker thread.
- **Batch inference.** Queue any subset of a dataset, with per-case and total timers,
  progress, cancel, and skip-if-already-done.
- **Result provenance.** Every prediction is written alongside a manifest recording the
  model, checkpoint epoch, validation Dice, SHA-256 of the weights, and timestamp.
- **Run history.** Every attempt, including failures, is appended to a CSV you can export.
- **3D Slicer viewing.** Opens any saved prediction over its source scan with surface
  smoothing disabled, so you see the model's actual voxel output rather than a
  cosmetically smoothed surface.

## Requirements

- Python 3.10 or newer
- A MONAI Label installation with a trained segmentation model:
  https://github.com/Project-MONAI/MONAILabel
- 3D Slicer for viewing, optional: https://www.slicer.org/
- A CUDA GPU for the server

## Install

    git clone https://github.com/21Kart28/unet-inference-client.git
    cd unet-inference-client
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

For development, `pip install -e ".[dev]"` adds pytest. Run the tests with `pytest`.

## Run

    unet-client

`python -m unet_inference_client` does the same.

On first launch, open the Settings and Server tabs and point the app at your MONAI Label
installation, dataset, and results folder. Settings persist to
~/.config/unet-inference-client/settings.json

## Dataset layout

The MONAI Label datastore expects one volume file per case at the top level of the
studies folder:

    studies/
      case_001.nrrd
      case_002.nrrd
      labels/
        final/
          case_001.nrrd
          case_002.nrrd

DICOM series folders are not read directly. Convert them to NRRD or NIfTI first.

## Notes

Built and tested against a DynUNet segmentation model served by MONAI Label 0.8.5.
Nothing in the code is specific to one organ or dataset. The model name, server URL,
and all paths are configurable.

## License

MIT
