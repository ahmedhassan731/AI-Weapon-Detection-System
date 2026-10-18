"""
Weapon & Threat Detection System
Supports Images, Batch Video Processing, and Live Webcam feeds using YOLOv8.
Model: Fine-tuned YOLOv8n from Hugging Face (akhil0238/Weapon_Detection)
Classes: Gun, Explosive, Grenade, Knife
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import urllib.request

# Global configuration defaults
HF_REPO = "akhil0238/Weapon_Detection"
HF_WEIGHTS_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/weights/best.pt"
DEFAULT_WEIGHTS = Path("models/best.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/detections")
DEFAULT_LOG_FILE = Path("logs/detections.csv")


def ensure_model_weights(weights_path: Path) -> Path:
    """Ensure weights file exists. If missing or 0 bytes, download from Hugging Face."""
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if not weights_path.exists() or weights_path.stat().st_size == 0:
        print(f"[*] Model weights not found at '{weights_path}'.")
        print(f"[*] Downloading pretrained weights from Hugging Face ({HF_REPO})...")
        try:
            urllib.request.urlretrieve(HF_WEIGHTS_URL, weights_path)
            print(f"[✓] Successfully downloaded weights to {weights_path} ({weights_path.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"[!] Error downloading from Hugging Face: {e}")
            print("[*] Falling back to default 'yolov8n.pt'...")
            return Path("yolov8n.pt")
    return weights_path


def log_detection(csv_path: Path, source_name: str, class_name: str, conf: float, bbox: list):
    """Append a detection event to the CSV log file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "source", "class_name", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, source_name, class_name, f"{conf:.4f}", *bbox])


def process_image(model, image_path: Path, conf: float, output_dir: Path, log_file: Path):
    """Run detection on a single image."""
    import cv2
    print(f"\n[*] Processing image: {image_path}")
    results = model.predict(source=str(image_path), conf=conf, verbose=False)

    for result in results:
        annotated_img = result.plot()
        boxes = result.boxes
        detected_count = len(boxes) if boxes is not None else 0

        print(f"[+] Detections in {image_path.name}: {detected_count}")

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = model.names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0].item())
                coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]

                print(f"    - Threat: {class_name.upper()} | Confidence: {confidence*100:.1f}% | Box: {coords}")
                log_detection(log_file, image_path.name, class_name, confidence, coords)

        out_path = output_dir / f"detected_{image_path.name}"
        cv2.imwrite(str(out_path), annotated_img)
        print(f"[✓] Annotated image saved to: {out_path}")


def process_video_batches(model, video_path: Path, conf: float, output_dir: Path, log_file: Path, batch_size: int, device):
    """Process video file using batch inference for maximum speed."""
    import cv2
    from tqdm import tqdm

    print(f"\n[*] Processing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file '{video_path}'")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_file = output_dir / f"detected_{video_path.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_file), fourcc, fps, (frame_width, frame_height))

    frames_batch = []
    total_threats_logged = 0

    def infer_and_write(batch):
        nonlocal total_threats_logged
        if not batch:
            return
        results = model(batch, conf=conf, device=device, verbose=False)
        for result in results:
            annotated_frame = result.plot()
            out.write(annotated_frame)

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = model.names.get(cls_id, str(cls_id))
                    confidence = float(box.conf[0].item())
                    coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]
                    log_detection(log_file, video_path.name, class_name, confidence, coords)
                    total_threats_logged += 1

    with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frames_batch.append(frame)

            if len(frames_batch) == batch_size:
                infer_and_write(frames_batch)
                pbar.update(len(frames_batch))
                frames_batch = []

    # Process any remaining frames
    if frames_batch:
        infer_and_write(frames_batch)
        pbar.update(len(frames_batch))

    cap.release()
    out.release()
    print(f"\n[✓] Video processing complete!")
    print(f"[✓] Output saved to: {out_file}")
    print(f"[✓] Total threat detections logged: {total_threats_logged}")


def process_webcam(model, cam_index: int, conf: float, log_file: Path, device):
    """Run real-time webcam detection with live video window."""
    import cv2
    print(f"\n[*] Starting Webcam (Camera Index: {cam_index})...")
    print("[*] Press 'q' or 'ESC' in the video window to quit.")
    
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[!] Error: Could not access webcam at index {cam_index}")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame from camera.")
            break

        results = model(frame, conf=conf, device=device, verbose=False)
        annotated_frame = results[0].plot()

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0.0
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Log detections
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = model.names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0].item())
                coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]
                log_detection(log_file, f"webcam_{cam_index}", class_name, confidence, coords)

        cv2.imshow("Weapon Detection System (Press 'q' to Exit)", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Webcam session ended.")


def main():
    parser = argparse.ArgumentParser(description="AI Weapon & Threat Detection System")
    parser.add_argument("--source", type=str, default="test_images/sample.jpg", 
                        help="Path to image, video file, folder, or webcam index (e.g. 0)")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), 
                        help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.4, 
                        help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--batch-size", type=int, default=16, 
                        help="Batch size for video processing")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), 
                        help="Output directory for annotated media")
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG_FILE), 
                        help="Path to CSV log file")
    parser.add_argument("--webcam", action="store_true", 
                        help="Shortcut to run live webcam detection")

    args = parser.parse_args()

    # Verify dependencies
    try:
        import torch
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        print(f"[!] Missing required library: {e}")
        print("[!] Run: pip install -r requirements.txt")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = f"GPU ({torch.cuda.get_device_name(0)})" if device == 0 else "CPU"
    print(f"[*] Using Computing Device: {device_name}")

    # Prepare paths
    weights_path = ensure_model_weights(Path(args.weights))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log)

    # Load YOLO Model
    print(f"[*] Loading YOLO model from '{weights_path}'...")
    model = YOLO(str(weights_path))
    print(f"[✓] Model loaded successfully! Classes: {model.names}")

    # Determine input type
    if args.webcam or args.source.isdigit():
        cam_id = int(args.source) if args.source.isdigit() else 0
        process_webcam(model, cam_id, args.conf, log_file, device)
        return

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"[!] Error: Source path '{source_path}' does not exist.")
        sys.exit(1)

    # Check file extensions
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix in video_extensions:
            process_video_batches(model, source_path, args.conf, output_dir, log_file, args.batch_size, device)
        elif suffix in image_extensions:
            process_image(model, source_path, args.conf, output_dir, log_file)
        else:
            print(f"[!] Unsupported file format: {suffix}")
    elif source_path.is_dir():
        all_images = [p for p in source_path.iterdir() if p.suffix.lower() in image_extensions]
        all_videos = [p for p in source_path.iterdir() if p.suffix.lower() in video_extensions]
        print(f"[*] Found {len(all_images)} image(s) and {len(all_videos)} video(s) in '{source_path}'")
        for img_path in all_images:
            process_image(model, img_path, args.conf, output_dir, log_file)
        for vid_path in all_videos:
            process_video_batches(model, vid_path, args.conf, output_dir, log_file, args.batch_size, device)


if __name__ == "__main__":
    main()
