# Smart PPE Detection System

## Overview

The Smart PPE Detection System is an AI-driven computer vision project designed to automatically detect and monitor the use of Personal Protective Equipment (PPE) in industrial and construction environments.  
It combines multiple deep learning modules for person detection, face recognition, tracking, and PPE compliance verification, ensuring worker safety through real-time visual analytics.

The system uses a fine-tuned YOLOv8 model specifically trained to identify key PPE elements such as helmets, safety vests, gloves, and boots, and to detect missing PPE instances with high precision and recall.

---

## System Architecture

### 1. Detection Modules

**Person Detection**  
Utilizes a pre-trained YOLOv8 model to detect human figures in an image or video stream.

**PPE Detection**  
Employs a fine-tuned YOLOv8 model trained on a custom PPE dataset to recognize and verify the presence of essential safety equipment.

**Face Recognition and Tracking**  
Implements face detection, alignment, and embedding-based recognition to identify employees and track them across frames.

**Re-Identification (ReID)**  
Associates detected individuals across multiple cameras or frames, maintaining identity consistency during video analysis.

### 2. Database Layer

**employee_db.py**  
Stores employee information including ID, facial embeddings, and metadata for recognition.

**log_db.py**  
Maintains logs of detected PPE violations and other events for audit and reporting.

### 3. Configuration Files

**config/config.yaml**  
Defines model paths, detection thresholds, and hardware settings.

**config/ppe.yaml**  
Specifies the PPE categories, class labels, and visualization parameters.

---

## Model Training Results

The YOLOv8 PPE model was fine-tuned on a curated dataset containing annotated images of key safety items.  
Training was performed under standard YOLOv8 configuration with early stopping, image augmentation, and mixed precision enabled.

| Parameter | Value |
|------------|--------|
| **Model** | YOLOv8 (fine-tuned) |
| **Training Run** | runs/detect/train7 |
| **Dataset** | Custom PPE dataset (helmet, vest, gloves, boots) |
| **mAP@0.5** | 56.1% |
| **Precision** | 0.90 |
| **Recall** | 0.83 |
| **Classes** | helmet, vest, gloves, boots |
| **Result** | The model effectively identifies missing PPE instances with high detection confidence and strong balance between precision and recall. |

---

## Installation

### Requirements

Install dependencies from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- ultralytics (YOLOv8 framework)  
- torch, torchvision  
- opencv-python  
- numpy, pandas, PyYAML  
- sqlite3  
- face_recognition  
- scikit-learn  

---

## Usage

### 1. PPE Detection on Images

Run the model to detect PPE elements on a folder of images:

```bash
python main.py --source data/images --task ppe
```

### 2. Video Detection and Tracking

Execute the detection pipeline on video files:

```bash
python main.py --source data/videos --task video
```

### 3. Employee Face Registration

Register new employees for facial recognition:

```bash
python tools/register_faces_to_db.py --image_dir data/employees
```

---

## Project Structure

```
smart_ppe_system/
│
├── config/              # Configuration files
├── database/            # Employee and log database modules
├── models/              # Detection, recognition, and tracking models
├── scripts/             # Dataset preparation and conversion utilities
├── tools/               # Face registration and helper scripts
├── tests/               # Test scripts for model validation
└── main.py              # Main entry point for the application
```

---

## Example Results

The fine-tuned YOLOv8 model accurately detects PPE components under varying lighting and environmental conditions.  
When missing PPE items are detected (e.g., no helmet or missing gloves), the system flags the instance and logs the event into the database for further review.

---

## Future Work

- Integration with edge devices (e.g., Jetson Nano, Raspberry Pi) for on-site inference.  
- Development of a real-time dashboard with visual alerts and violation reporting.  
- Improved mAP through dataset expansion and synthetic augmentation.  
- Implementation of multi-camera identity tracking and temporal PPE compliance analytics.

---

## Author

**Pham Tuan**  
Email: 1604phamtuan@gmail.com  

Project: *Smart PPE Detection System (YOLOv8 Fine-tuned for PPE Safety)*
