# 🛡️ AI Weapon & Threat Detection System

A complete real-time weapon and threat detection system using fine-tuned **YOLOv8** on surveillance datasets.
Pretrained model: [akhil0238/Weapon_Detection](https://huggingface.co/akhil0238/Weapon_Detection)

### 🎯 Detected Threat Classes:
1. **Gun** (Pistols, rifles, firearms)
2. **Knife** (Knives, daggers, bladed weapons)
3. **Grenade** (Hand grenades, explosives)
4. **Explosive** (Explosion hazards, fire)

---

## 📁 Directory Structure

```text
Weapon Detection/
│
├── app.py                  # Main inference script (Images, Videos, Webcam)
├── requirements.txt        # All required Python packages
├── README.md               # Complete documentation
│
├── models/
│   └── best.pt             # Pretrained YOLOv8 weights (auto-downloaded)
│
├── outputs/
│   └── detections/         # Saved detections (images & videos with bounding boxes)
│
├── logs/
│   └── detections.csv      # Real-time CSV detection audit logs
│
└── test_images/
    └── sample.jpg          # Sample test image
```

---

## 🚀 Setup & Installation

### Step 1: Install Dependencies
Open PowerShell / Command Prompt in this folder and run:

```bash
pip install -r requirements.txt
```

*(Note: PyTorch, OpenCV, Ultralytics, Hugging Face Hub, and tqdm will be installed).*

### Step 2: Model Weights
Model weights (`models/best.pt`) are already placed in `models/best.pt`.
If deleted, `app.py` will automatically download them from Hugging Face on the first run.

---

## 💻 How to Run

### 🌟 Simple Gradio Web Frontend (Recommended)
```bash
python gradio_app.py
```
Opens an interactive web dashboard in your browser (`http://127.0.0.1:7860`) where you can:
- **Live Webcam Detection**: Real-time continuous in-browser stream or 1-click ultra-fast Native OpenCV camera window.
- **Image Detection**: Upload images or capture webcam snapshots to detect weapons.
- **Video Detection**: Upload surveillance videos to analyze and generate annotated MP4 outputs.
- **Confidence Slider**: Adjust detection sensitivity dynamically.
- **Audit Logs**: Inspect real-time threat audit logs directly in the UI.

### 1. CLI: Detect on a Single Image
```bash
python app.py --source test_images/sample.jpg
```
Output image with bounding boxes will be saved in `outputs/detections/`.

### 2. Detect on a Folder of Images
```bash
python app.py --source test_images/
```

### 3. Detect on a Video (with Batch Processing)
```bash
python app.py --source input_video.mp4 --conf 0.4 --batch-size 16
```
Annotated video with bounding boxes will be saved to `outputs/detections/detected_input_video.mp4`.

### 4. Real-Time Live Webcam Detection
```bash
python app.py --webcam
# OR
python app.py --source 0
```
- Shows live bounding boxes, threat labels, confidence score, and real-time FPS.
- Press **'q'** or **ESC** to stop the webcam feed.

---

## 📊 CSV Logging Format (`logs/detections.csv`)

Every detected threat is automatically recorded in real-time:

| timestamp | source | class_name | confidence | bbox_x1 | bbox_y1 | bbox_x2 | bbox_y2 |
|-----------|--------|------------|------------|---------|---------|---------|---------|
| 2026-10-14 12:45:10 | sample.jpg | knife | 0.8845 | 120.5 | 45.2 | 350.1 | 480.0 |
| 2026-10-14 12:45:12 | input_video.mp4 | gun | 0.9210 | 210.0 | 95.0 | 410.5 | 320.4 |

---

## ⚙️ Advanced Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `test_images/sample.jpg` | Path to image, video, directory, or camera index |
| `--weights` | `models/best.pt` | Path to model weights file |
| `--conf` | `0.4` | Detection confidence threshold (0.0 to 1.0) |
| `--batch-size` | `16` | Batch size for fast video inference |
| `--output` | `outputs/detections` | Output directory for annotated media |
| `--log` | `logs/detections.csv` | File path for CSV logs |
| `--webcam` | `False` | Run live webcam detection |
