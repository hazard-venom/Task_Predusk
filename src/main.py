from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-object detection and persistent ID tracking using YOLO and DeepSORT."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the input video. Defaults to the first .mp4 file in input/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "tracked_output.mp4",
        help="Path to the annotated output video.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model name or path.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["person"],
        help="Target class names to track, for example: person sports ball car.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display a live preview window while processing.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=20,
        help="Number of previous center points to keep for each track trail.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device to use, for example: cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def resolve_input_video(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        candidate = explicit_path if explicit_path.is_absolute() else PROJECT_ROOT / explicit_path
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Input video not found: {candidate}")

    video_files = sorted(INPUT_DIR.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No .mp4 file found in {INPUT_DIR}")
    return video_files[0]


def resolve_output_path(output_path: Path) -> Path:
    candidate = output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def build_heatmap_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_heatmap.png")


def build_per_id_heatmap_dir(output_path: Path) -> Path:
    heatmap_dir = output_path.with_name(f"{output_path.stem}_per_id_heatmaps")
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    return heatmap_dir


def resolve_target_class_ids(model: YOLO, class_names: Iterable[str]) -> set[int]:
    names = model.names
    normalized = {str(name).lower(): idx for idx, name in names.items()}
    requested = {name.lower() for name in class_names}
    matched = {normalized[name] for name in requested if name in normalized}

    if not matched:
        available = ", ".join(str(name) for name in names.values())
        requested_text = ", ".join(sorted(requested))
        raise ValueError(
            f"None of the requested classes were found: {requested_text}. Available classes: {available}"
        )

    return matched


def draw_track_trail(
    frame: cv2.typing.MatLike,
    history: deque[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    if len(history) < 2:
        return

    points = list(history)
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], color, 2)


def draw_overlay(
    frame: cv2.typing.MatLike,
    frame_index: int,
    total_frames: int,
    active_tracks: int,
    total_unique_ids: int,
) -> None:
    overlay_lines = [
        f"Frame: {frame_index}/{total_frames}" if total_frames else f"Frame: {frame_index}",
        f"Active tracks: {active_tracks}",
        f"Unique IDs seen: {total_unique_ids}",
    ]

    for index, text in enumerate(overlay_lines):
        y_position = 30 + index * 28
        cv2.putText(
            frame,
            text,
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )


def save_heatmap(heatmap: np.ndarray, output_path: Path) -> Path:
    normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = normalized.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_path = build_heatmap_path(output_path)
    cv2.imwrite(str(heatmap_path), colored_heatmap)
    return heatmap_path


def save_per_id_heatmaps(
    track_points: dict[int, list[tuple[int, int]]],
    frame_width: int,
    frame_height: int,
    output_path: Path,
) -> list[Path]:
    heatmap_dir = build_per_id_heatmap_dir(output_path)
    saved_paths: list[Path] = []

    for track_id in sorted(track_points):
        points = track_points[track_id]
        if not points:
            continue

        canvas = np.zeros((frame_height, frame_width), dtype=np.float32)
        for center in points:
            cv2.circle(canvas, center, 8, 1.0, thickness=-1)

        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=11, sigmaY=11)
        normalized = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = normalized.astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        label_height = 50
        labeled = np.zeros((frame_height + label_height, frame_width, 3), dtype=np.uint8)
        labeled[label_height:, :, :] = colored_heatmap
        cv2.putText(
            labeled,
            f"Track ID {track_id}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            labeled,
            "Blue = lower presence, Red = higher presence",
            (20, frame_height + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        heatmap_path = heatmap_dir / f"track_id_{track_id}.png"
        cv2.imwrite(str(heatmap_path), labeled)
        saved_paths.append(heatmap_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    input_path = resolve_input_video(args.input)
    output_path = resolve_output_path(args.output)

    model = YOLO(args.model)
    target_class_ids = resolve_target_class_ids(model, args.classes)
    tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    track_history: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=args.trail_length)
    )
    unique_track_ids: set[int] = set()
    track_points: dict[int, list[tuple[int, int]]] = defaultdict(list)
    processed_frames = 0

    print(f"Input video: {input_path}")
    print(f"Output video: {output_path}")
    print(f"Tracking classes: {', '.join(args.classes)}")
    print(f"Inference device: {args.device}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False, device=args.device)[0]
        detections = []

        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = detection
            class_id = int(cls)
            if class_id not in target_class_ids:
                continue

            width = x2 - x1
            height = y2 - y1
            label = str(model.names[class_id])
            detections.append(([x1, y1, width, height], conf, label))

        tracks = tracker.update_tracks(detections, frame=frame)
        active_tracks = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

            active_tracks += 1
            track_id = track.track_id
            unique_track_ids.add(track_id)
            label = track.get_det_class() or "object"
            left, top, right, bottom = map(int, track.to_ltrb())
            center = ((left + right) // 2, (top + bottom) // 2)
            track_history[track_id].append(center)
            track_points[track_id].append(center)

            color = (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            draw_track_trail(frame, track_history[track_id], color)
            cv2.putText(
                frame,
                f"{label} | ID {track_id}",
                (left, max(25, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        processed_frames += 1
        draw_overlay(frame, processed_frames, total_frames, active_tracks, len(unique_track_ids))

        if processed_frames % 30 == 0:
            if total_frames:
                print(f"Processed {processed_frames}/{total_frames} frames")
            else:
                print(f"Processed {processed_frames} frames")

        out.write(frame)

        if args.show:
            cv2.imshow("Multi-Object Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Completed. Annotated video saved at: {output_path}")
    print(f"Unique IDs tracked: {len(unique_track_ids)}")
    per_id_heatmap_paths = save_per_id_heatmaps(track_points, frame_width, frame_height, output_path)
    heatmap_dir = build_per_id_heatmap_dir(output_path)
    print(f"Per-ID heatmaps saved in: {heatmap_dir}")
    print(f"Per-ID heatmap files generated: {len(per_id_heatmap_paths)}")


if __name__ == "__main__":
    main()
