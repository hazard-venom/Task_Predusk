# Multi-Object Detection and Persistent ID Tracking

This project implements a computer vision pipeline for multi-object detection and persistent ID tracking in public sports or event footage. It uses YOLOv8 for object detection and DeepSORT for assigning stable IDs across frames.

## Features

- Detects multiple moving subjects in a video
- Assigns persistent IDs across frames
- Writes an annotated output video with bounding boxes and IDs
- Overlays active-track and unique-object counts on the output video
- Saves separate per-ID movement heatmaps for each unique tracked subject
- Optional live preview during processing
- Supports GPU inference with CUDA when a CUDA-enabled PyTorch build is installed
- Supports configurable target classes such as `person`, `sports ball`, `car`
- Draws short trajectory trails to make motion easier to inspect

## Project Structure

```text
Task_preddusk/
|-- input/
|-- output/
|-- src/
|   `-- main.py
|-- requirements.txt
|-- README.md
`-- technical_report.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Dependencies

- `ultralytics`
- `opencv-python`
- `deep-sort-realtime`
- `numpy`
- `torch`

## GPU Support

The pipeline supports GPU inference through PyTorch and Ultralytics.

- Default behavior: the script selects `cuda` when `torch.cuda.is_available()` is `True`
- Manual override: use `--device cuda`, `--device cuda:0`, or `--device cpu`
- Important: a CPU-only PyTorch build will force the pipeline to run on CPU even if your machine has an NVIDIA GPU

Example CUDA check:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If CUDA is not available in your environment, install a CUDA-enabled PyTorch build in the same virtual environment before running the pipeline on GPU.

Current environment note for this repository:

- Host GPU detected: `NVIDIA GeForce GTX 1650`
- Current project virtual environment state: `torch 2.11.0+cpu`
- Result: the script will still run on CPU until PyTorch is reinstalled with CUDA support in `myenv`

## How To Run

If you want the script to pick the first `.mp4` file from the `input/` folder automatically:

```powershell
python .\src\main.py
```

To show the live preview window while processing:

```powershell
python .\src\main.py --show
```

To force GPU execution:

```powershell
python .\src\main.py --device cuda --show
```

For this repository's current virtual environment, GPU execution will not activate until the CPU-only PyTorch build is replaced.

Suggested fix:

```powershell
.\myenv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\myenv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then verify:

```powershell
.\myenv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

To use a specific video and track a different class mix:

```powershell
python .\src\main.py --input .\input\your_video.mp4 --classes person "sports ball" --show
```

To write the output file to a custom location:

```powershell
python .\src\main.py --output .\output\match_annotated.mp4
```

## Pipeline Summary

1. Load a YOLOv8 model for frame-level object detection.
2. Filter detections to the requested target classes.
3. Convert detections into DeepSORT tracking inputs.
4. Assign stable IDs across frames.
5. Render bounding boxes, class labels, track IDs, and short motion trails.
6. Overlay active object count and unique ID count on each frame.
7. Accumulate tracked center points separately for each unique track ID.
8. Save the annotated result as an `.mp4` video and one heatmap `.png` per ID.

## Model And Tracker Choice

- Detector: `YOLOv8n`
  - Fast, lightweight, and easy to run on CPU for a small assignment project
  - Can also be accelerated on GPU when CUDA-enabled PyTorch is installed
  - Strong enough for person-centric public event footage
- Tracker: `DeepSORT`
  - Maintains track identity across frames
  - Handles short occlusions and moderate camera motion better than naive frame-to-frame matching

## Assumptions

- The input video is public and legally shareable for demonstration.
- The most important subjects belong to one or more COCO classes available in YOLOv8.
- A lightweight detector is preferred over a heavier, slower model for quick experimentation.
- The runtime environment has the correct CUDA-enabled PyTorch build if GPU acceleration is expected.

## Limitations

- ID switches can still happen during long occlusion or dense crowd overlap.
- Small or blurry subjects may be missed by `yolov8n`.
- Fast camera pans and extreme zoom changes can reduce tracking quality.
- For sport-specific analysis, a domain-tuned model would outperform a generic COCO detector.
- GPU execution depends on environment setup; having CUDA drivers alone is not enough if PyTorch was installed as a CPU-only build.
- In the current checked environment for this repo, CUDA-capable hardware is present but the active virtual environment contains `torch 2.11.0+cpu`, so inference remains on CPU until PyTorch is reinstalled with CUDA support.




