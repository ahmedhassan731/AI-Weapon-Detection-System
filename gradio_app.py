"""
AI Weapon & Threat Detection System - Gradio Web Frontend
Simple, clean UI for detecting weapons (Gun, Knife, Grenade, Explosive) in Images, Live Webcam, and Videos.
"""

import csv
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Paths & Defaults
HF_REPO = "akhil0238/Weapon_Detection"
HF_WEIGHTS_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/weights/best.pt"
MODEL_PATH = Path("models/best.pt")
LOG_PATH = Path("logs/detections.csv")
OUTPUT_DIR = Path("outputs/detections")

# Ensure required directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Global process tracker for native OpenCV webcam
webcam_process = None


def ensure_model_weights(weights_path: Path) -> Path:
    """Download pretrained weights if not present."""
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if not weights_path.exists() or weights_path.stat().st_size == 0:
        print(f"[*] Downloading weights from {HF_WEIGHTS_URL}...")
        try:
            urllib.request.urlretrieve(HF_WEIGHTS_URL, weights_path)
            print("[✓] Model weights downloaded.")
        except Exception as e:
            print(f"[!] Download failed: {e}. Falling back to yolov8n.pt")
            return Path("yolov8n.pt")
    return weights_path


# Load model at startup
weights = ensure_model_weights(MODEL_PATH)
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(str(weights))


def log_detection(source_name: str, class_name: str, conf: float, bbox: list):
    """Log detection event to CSV."""
    is_new = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    with open(LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "source", "class_name", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, source_name, class_name, f"{conf:.4f}", *bbox])


def detect_image(input_img, conf_threshold):
    """Run detection on single uploaded image or webcam snapshot."""
    if input_img is None:
        return None, "⚠️ Please upload or capture an image.", None

    results = model.predict(source=input_img, conf=conf_threshold, device=device, verbose=False)
    result = results[0]

    # Convert annotated image from BGR to RGB for Gradio display
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0].item())
            coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]

            detections.append({
                "Threat Class": class_name.upper(),
                "Confidence": f"{confidence * 100:.1f}%",
                "Bounding Box [x1, y1, x2, y2]": str(coords)
            })
            log_detection("gradio_image", class_name, confidence, coords)

    if detections:
        threat_counts = {}
        for d in detections:
            c = d["Threat Class"]
            threat_counts[c] = threat_counts.get(c, 0) + 1
        summary_str = ", ".join([f"{count}x {cls}" for cls, count in threat_counts.items()])
        status = f"🚨 **THREAT DETECTED!** Found: **{summary_str}**"
        df = pd.DataFrame(detections)
    else:
        status = "✅ **SAFE:** No weapon or threat detected."
        df = pd.DataFrame(columns=["Threat Class", "Confidence", "Bounding Box [x1, y1, x2, y2]"])

    return annotated_rgb, status, df


def detect_live_frame(frame, conf_threshold):
    """Process a single frame from the browser streaming webcam."""
    if frame is None:
        return None, "📷 Start webcam to begin live threat monitoring."

    # Predict frame
    results = model.predict(source=frame, conf=conf_threshold, device=device, verbose=False)
    result = results[0]

    # Convert BGR to RGB
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        threats = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0].item())
            coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]
            threats.append(f"{class_name.upper()} ({confidence * 100:.0f}%)")
            log_detection("live_webcam", class_name, confidence, coords)

        status = f"🚨 **ALERT! Threat Detected in Webcam:** {', '.join(threats)}"
    else:
        status = "🟢 **MONITORING:** Area is clear. No weapons detected."

    return annotated_rgb, status


def start_native_webcam(conf_val):
    """Launch native OpenCV webcam window for full hardware FPS."""
    global webcam_process
    if webcam_process is not None and webcam_process.poll() is None:
        return "⚠️ OpenCV Webcam window is already running! (Press 'q' or 'ESC' in the webcam window to exit)."

    try:
        cmd = [sys.executable, "app.py", "--webcam", "--conf", str(conf_val)]
        webcam_process = subprocess.Popen(cmd)
        return "✅ **Native OpenCV Webcam launched!** Check your taskbar/screen for the video window.\n\n*(Press 'q' or 'ESC' on the camera window to stop).* "
    except Exception as e:
        return f"❌ Failed to start webcam: {e}"


def stop_native_webcam():
    """Stop the native OpenCV webcam window."""
    global webcam_process
    if webcam_process is not None and webcam_process.poll() is None:
        webcam_process.terminate()
        webcam_process = None
        return "🛑 Native OpenCV Webcam stopped."
    return "ℹ️ No native webcam window is currently active."


def detect_video(video_path, conf_threshold, progress=gr.Progress()):
    """Run detection on an uploaded video file."""
    if not video_path:
        return None, "⚠️ Please upload a video file first."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video file."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_file = OUTPUT_DIR / f"gradio_detected_{Path(video_path).stem}_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    detected_counts = {}
    total_threats = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf_threshold, device=device, verbose=False)
        annotated = results[0].plot()
        out.write(annotated)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                class_name = model.names.get(cls_id, str(cls_id))
                confidence = float(box.conf[0].item())
                coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]
                detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
                total_threats += 1
                log_detection(Path(video_path).name, class_name, confidence, coords)

        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            progress(frame_idx / total_frames, desc=f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    if total_threats > 0:
        summary_str = ", ".join([f"{count}x {cls.upper()}" for cls, count in detected_counts.items()])
        status = f"🚨 **Analysis Complete:** {total_threats} threats detected ({summary_str})."
    else:
        status = "✅ **Analysis Complete:** No threats detected throughout the video."

    return str(out_file), status


def load_logs():
    """Load recent detection audit logs from CSV."""
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "source", "class_name", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
    try:
        df = pd.read_csv(LOG_PATH)
        return df.tail(50).iloc[::-1]  # Most recent first, up to 50 rows
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# Build Gradio UI
with gr.Blocks(title="AI Weapon Detection System") as demo:
    gr.Markdown(
        """
        # 🛡️ AI Weapon & Threat Detection System
        Real-time detection of firearms, bladed weapons, grenades, and explosives using fine-tuned **YOLOv8**.
        
        *Detected Threat Classes:* **Gun**, **Knife**, **Grenade**, **Explosion**
        """
    )

    with gr.Tabs():
        # TAB 1: Live Webcam (Real-time)
        with gr.TabItem("📹 Live Webcam"):
            gr.Markdown("### Real-Time Live Camera Detection")
            with gr.Row():
                with gr.Column():
                    webcam_stream_in = gr.Image(sources=["webcam"], streaming=True, label="Live Camera Feed")
                    webcam_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.40, step=0.05, label="Confidence Threshold")
                with gr.Column():
                    webcam_stream_out = gr.Image(label="Annotated Live Detection")
                    webcam_status = gr.Markdown(value="🟢 Turn on your webcam above to begin real-time detection.")

            webcam_stream_in.stream(
                fn=detect_live_frame,
                inputs=[webcam_stream_in, webcam_conf],
                outputs=[webcam_stream_out, webcam_status],
                stream_every=0.1,
                show_progress="hidden"
            )

            gr.Markdown("---")
            gr.Markdown("#### ⚡ Desktop High-FPS Webcam Mode")
            gr.Markdown("Run a direct native window at full 30+ FPS directly on your hardware:")
            with gr.Row():
                native_start_btn = gr.Button("🚀 Open Native Camera Window", variant="primary")
                native_stop_btn = gr.Button("⏹️ Stop Native Camera", variant="stop")
            native_status = gr.Markdown("")

            native_start_btn.click(fn=start_native_webcam, inputs=[webcam_conf], outputs=native_status)
            native_stop_btn.click(fn=stop_native_webcam, outputs=native_status)

        # TAB 2: Image Detection
        with gr.TabItem("📷 Image Detection"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="Upload Image / Take Snapshot")
                    img_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.40, step=0.05, label="Confidence Threshold")
                    img_btn = gr.Button("🔍 Detect Threats", variant="primary")

                with gr.Column():
                    img_output = gr.Image(type="numpy", label="Detection Result")
                    img_status = gr.Markdown(value="Waiting for image...")
                    img_table = gr.DataFrame(
                        headers=["Threat Class", "Confidence", "Bounding Box [x1, y1, x2, y2]"],
                        label="Detected Threat Details"
                    )

            img_btn.click(
                fn=detect_image,
                inputs=[img_input, img_conf],
                outputs=[img_output, img_status, img_table]
            )

        # TAB 3: Video Detection
        with gr.TabItem("🎥 Video Detection"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video File")
                    vid_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.40, step=0.05, label="Confidence Threshold")
                    vid_btn = gr.Button("🎬 Analyze Video", variant="primary")

                with gr.Column():
                    vid_output = gr.Video(label="Annotated Output Video")
                    vid_status = gr.Markdown(value="Waiting for video...")

            vid_btn.click(
                fn=detect_video,
                inputs=[vid_input, vid_conf],
                outputs=[vid_output, vid_status]
            )

        # TAB 4: Detection Audit Logs
        with gr.TabItem("📊 Detection Audit Logs"):
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Logs", variant="secondary")
            log_table = gr.DataFrame(value=load_logs, label="Recent Detection Logs (CSV)", every=5)
            refresh_btn.click(fn=load_logs, outputs=log_table)

    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: gray; font-size: 0.9em;">
        AI Weapon Detection System • Model: YOLOv8n fine-tuned on Weapon Detection • Real-time Threat Auditing
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="red"),
        inbrowser=True
    )
