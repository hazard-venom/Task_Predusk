# Technical Report: Multi-Object Detection and Persistent ID Tracking

## 1. Overview

This project implements a multi-object tracking pipeline for public sports or event footage. The system detects relevant subjects in each frame and assigns persistent IDs so that the same subject can be followed across the full video. The output is an annotated video with bounding boxes, class labels, track IDs, and short trajectory trails. Separate per-ID heatmaps are also generated to show where each tracked subject spent most of its time in the image.

## 2. Detector Used

The detector used is `YOLOv8n` from Ultralytics.

Reasons for selection:

- It is lightweight and fast enough to run on common laptops.
- It can run on either CPU or GPU depending on the PyTorch installation and selected device.
- It provides strong baseline detection quality for COCO classes such as `person`, `sports ball`, and `car`.
- It is simple to integrate into a Python pipeline and suitable for rapid experimentation.

## 3. Tracking Algorithm Used

The tracker used is `DeepSORT` via the `deep-sort-realtime` package.

Reasons for selection:

- It supports online multi-object tracking with persistent IDs.
- It combines motion and appearance cues, which improves ID stability compared with naive IOU-only tracking.
- It handles short-term occlusion and partial visibility reasonably well for practical sports/event footage.

## 4. Why This Detector + Tracker Combination Was Chosen

YOLOv8 provides fast and accurate detections, while DeepSORT associates those detections across frames. This combination is a common practical baseline for multi-object tracking because it separates the problem into two reliable stages:

- frame-wise subject detection
- temporal identity association

This modular design makes the system easier to debug, tune, and extend.

The implementation also supports GPU inference through CUDA, which reduces frame processing time when a CUDA-enabled PyTorch build is available.

In the current project environment, the host machine has an NVIDIA GeForce GTX 1650 GPU and working NVIDIA drivers, but the active virtual environment contains a CPU-only PyTorch build (`torch 2.11.0+cpu`). Because of that, `torch.cuda.is_available()` returns `False` and the current run path stays on CPU until the environment is updated.

## 5. How ID Consistency Is Maintained

ID consistency is maintained by passing YOLO detections to DeepSORT on every frame. DeepSORT estimates object motion over time and matches new detections to existing tracks using tracking state and appearance information. As a result, the same subject typically keeps the same ID even when there is:

- moderate movement
- brief overlap with another subject
- short-term occlusion
- camera motion

Short trajectory trails are also drawn in the output to visually inspect whether track continuity is stable.

The pipeline also maintains a set of confirmed track IDs to estimate how many unique subjects were observed across the video. This value is displayed in the overlay and printed after processing completes. For interpretability, tracked center points are stored separately per ID and exported as individual heatmaps instead of a single combined density map.

## 6. Challenges Faced

- Occlusion between nearby subjects can cause temporary track loss.
- Similar-looking players or participants can cause ID switches.
- Motion blur and low-resolution footage reduce detection quality.
- Generic COCO models are not tailored for every sports domain.
- GPU acceleration can fail if the Python environment contains a CPU-only PyTorch build even when the machine has CUDA-capable hardware.
- This exact issue was observed in the current repository environment: CUDA hardware was present, but the active virtual environment did not include a CUDA-enabled PyTorch build.
- The per-ID heatmaps are image-plane based and do not compensate for perspective, so they should be interpreted as visual activity maps rather than true top-view density estimates.

## 7. Failure Cases Observed

Typical failure cases include:

- IDs switching after long overlap between subjects
- missed detections for small or distant subjects
- unstable tracking during fast camera pan or zoom
- incomplete tracking for classes not well represented in the pretrained detector

## 8. Possible Improvements

- Use a stronger detector such as `yolov8s` or a domain-specific fine-tuned model.
- Tune DeepSORT parameters for the selected sport or event.
- Add frame skipping and benchmarking for speed/accuracy tradeoff analysis.
- Evaluate multiple detector-tracker combinations and compare ID stability.
- Benchmark CPU versus GPU inference and report end-to-end FPS on the selected video.
- Replace the current CPU-only PyTorch installation in the project virtual environment with a CUDA-enabled build and rerun the pipeline on GPU.



