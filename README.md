# AI-Based Real-Time Forest Fire and Smoke Monitoring System Using Hybrid Deep Learning

## 1. Project Title
AI-Based Real-Time Forest Fire and Smoke Monitoring System Using Hybrid Deep Learning

## 2. Abstract
This project develops an AI-powered wildfire monitoring system that detects fire and smoke from satellite imagery, surveillance images, uploaded images, and live video streams. The system uses a hybrid deep-learning pipeline built around YOLO for object detection, EfficientNet for classification, and LSTM for temporal growth analysis. It filters false positives, estimates severity, maps event location, and raises early-warning alerts when risk crosses a configurable threshold.

The project is designed for a college demonstration and is intentionally modular. The primary working pipeline is YOLO + EfficientNet + LSTM. Other model types are included as optional experimental alternatives to compare performance on a normal computer.

## 3. Problem Statement
Early wildfire detection can reduce devastation, save habitats, and prevent large-scale destruction. Traditional satellite and camera monitoring often fail to distinguish genuine wildfire smoke from clouds, fog, haze, dust, sunlight glare, and industrial smoke. There is therefore a need for a real-time AI-based system that combines spatial detection and temporal behavior analysis.

## 4. Objectives
- Detect fire and smoke from images and videos.
- Reduce false positives caused by clouds, fog, dust, or sunlight glare.
- Estimate severity and growth risk.
- Perform temporal analysis for early-warning detection.
- Show results in a dashboard and alert history.
- Keep the application runnable in demo mode without a trained model.

## 5. Existing System
Existing wildfire monitoring systems often depend on manual inspection or simple threshold-based rules. These systems may not handle real-time video, false positives, or early growth warnings well.

## 6. Proposed System
The proposed system uses:
- YOLO for fire/smoke detection
- EfficientNet for local region classification
- LSTM for temporal wildfire progression
- False-positive filtering using visual and temporal confidence checks
- Severity estimation and growth probability prediction
- SQLite-backed alert logs and dashboard stats

## 7. Dataset
The project is designed for public NASA Earthdata datasets, particularly MODIS and VIIRS fire products.

Official dataset source:
https://www.earthdata.nasa.gov/

### Public dataset guidance
1. Visit NASA Earthdata.
2. Search for MODIS Thermal Anomalies / Fire or VIIRS fire products.
3. Download a manageable subset for training and validation.
4. Place files under the project dataset folders as described below.

### Folder structure
```text
forest_fire_ai/
  dataset/
    modis/
      2024/
    train/
    val/
    test/
```

### Recommended class folders
```text
forest_fire_ai/dataset/train/
  fire/
  smoke/
  no_fire/
  cloud/
  fog/
  dust/
```

If the NASA dataset uses different labels, use the preprocessing conversion script to map them into the required project classes.

The project includes the MODIS 2024 satellite collection under
`dataset/modis/2024/`. It contains 207 country-level CSV files extracted from
the supplied `data_set.zip`. These are tabular satellite observations for
geographic analysis, not image-class folders for model training.

> Important: Do not download a massive dataset automatically. The project includes clear instructions and a demo mode so it can run without a full dataset.

## 8. Dataset Preprocessing
The preprocessing pipeline includes:
- image resizing
- normalization
- augmentation
- noise reduction
- contrast enhancement
- duplicate image detection
- corrupted image detection
- class imbalance handling
- train/validation/test split
- leakage checks

The project supports common splits such as:
- 80/20 for train/test
- 70/15/15 for train/validation/test

## 9. AI Algorithms
### Model 1: YOLO + EfficientNet + LSTM (primary model)
- YOLO detects fire/smoke regions.
- EfficientNet classifies the detected region.
- LSTM analyzes temporal sequences to estimate risk growth.

### Model 2: CNN + BiLSTM + Attention
- CNN extracts image features.
- BiLSTM captures sequential patterns.
- Attention highlights key frames and areas.

### Model 3: Swin Transformer + CNN
- Swin Transformer extracts global visual features.
- CNN refines predictions and reduces false positives.

### Model 4: YOLO + Vision Transformer + LSTM
- YOLO detects candidate regions.
- ViT extracts high-level features.
- LSTM handles temporal growth analysis.

### Model 5: ConvLSTM + Attention
- ConvLSTM processes sequences of frames.
- Attention is used for early warning risk scoring.

## 10. System Architecture
```text
Camera / Satellite / Upload / Video
        ↓
Preprocessing
        ↓
YOLO Detection
        ↓
EfficientNet Classification
        ↓
LSTM Temporal Prediction
        ↓
False Positive Filtering
        ↓
Severity Estimation
        ↓
Alert Generation
        ↓
Dashboard + Map
```

## 11. Methodology
1. Load image or video input.
2. Preprocess data.
3. Detect objects with YOLO.
4. Crop the fire/smoke region.
5. Classify region with EfficientNet.
6. Analyze sequential changes with LSTM.
7. Merge visual and temporal evidence.
8. Estimate severity and affected area.
9. Store result in SQLite and show on dashboard.

## 12. Model Training
Use the training script:

```powershell
cd C:\Users\Sanjay S\forest_fire_ai
py .\scripts\train.py
```

The training script:
- detects GPU if available
- falls back to CPU if unavailable
- trains a scikit-learn `MLPClassifier` from labeled image folders
- saves `outputs/models/best_model.pkl`
- uses RGB image features compatible with the upload API
- returns `not_trained` instead of fabricating a checkpoint when images are absent

The image training dataset must be arranged as:

```text
dataset/train/fire/
dataset/train/smoke/
dataset/train/no_fire/
dataset/val/fire/
dataset/val/smoke/
dataset/val/no_fire/
dataset/test/fire/
dataset/test/smoke/
dataset/test/no_fire/
```

The supplied `dataset/modis/2024` files are real MODIS tabular satellite
observations and are used by the satellite nearby lookup. They are not labeled
camera images, so they cannot be used to train the image MLP or produce
image-classification metrics. The training command reports this honestly until
an image dataset is added.

Generate a real MODIS dataset report with:

```powershell
py .\scripts\modis_report.py
```

The image response also includes the nearest MODIS observation when GPS
coordinates are available. Its satellite risk score combines the observation
confidence (70%) and normalized fire radiative power (30%); no coordinates or
satellite values are fabricated.

## 13. Evaluation
The evaluation script produces:
- Accuracy
- Precision
- Recall
- F1-score
- confusion matrix
- ROC-AUC where applicable
- mAP for YOLO
- IoU
- inference speed

Run the image-classifier evaluation after training:

```powershell
py .\scripts\evaluate_image_model.py
```

The report is written to `outputs/image_evaluation_report.json`.

The evaluation report compares the five models and prints “Not evaluated” when a model has not been trained.

## 14. Results
This project intentionally does not fabricate results. Real model results must be produced by trained runs using actual dataset files. Until training is completed, the application runs in demo mode and clearly labels all predictions as demo outputs.

## 15. Installation
### Python environment
```powershell
cd C:\Users\Sanjay S\forest_fire_ai
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

### Frontend environment
```powershell
cd C:\Users\Sanjay S\forest_fire_ai\frontend
npm install
```

## 16. How to Run
### 16.1 Dataset (real labeled images)
```powershell
cd C:\Users\Sanjay S\forest_fire_ai
.\.venv\Scripts\Activate.ps1
python scripts/download_fire_dataset.py   # DeepQuestAI v1, ~3000 images, 70/15/15 split
python scripts/dedupe_splits.py           # remove cross-split duplicates (leakage)
```
Report: `outputs/dataset_report.json` (counts, corrupted, duplicates).

### 16.2 Training (real, reproducible)
```powershell
python scripts/train.py                   # MLP on 32x32 RGB features -> outputs/models/best_model.pkl
python scripts/evaluate_image_model.py    # test metrics -> outputs/image_evaluation_report.json
python scripts/verify_honest.py           # 32 honesty checks (must be 0 failed)
```
Real results (DeepQuestAI v1, 2931 images after dedupe):
train acc 0.7126 / macro-F1 0.7151 · val acc 0.6159 · **test acc 0.6636 / macro-F1 0.6665** (428 unseen).

### Backend
```powershell
cd C:\Users\Sanjay S\forest_fire_ai
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

### Frontend
```powershell
cd C:\Users\Sanjay S\forest_fire_ai\frontend
npm run dev
```

Open the browser at:
- Frontend: http://127.0.0.1:5173
- Backend API: http://127.0.0.1:8001/docs

## 17. API Documentation
Image analysis endpoints (`multipart/form-data` with `file`; optional `latitude`+`longitude` together):
- POST /api/detect/image — main analysis
- POST /upload-image, POST /analyze-image, POST /predict — aliases (same handler)
- POST /predict-sequence — always 501 (TEMPORAL MODEL NOT TRAINED; single images cannot forecast)
- GET /model-info, GET /model_info — architecture, classes, input spec, checkpoint paths
- GET /health — backend + model status
- GET /history?limit=50 — saved detections (SQLite)
- GET /history/{id} — one detection (404 if missing)
- DELETE /history/{id} — delete a detection (404 if missing)
- GET /api/alerts, GET /api/alerts/{id}, GET /api/statistics, GET /api/model-performance
- GET /temporal-status, GET /api/satellite/nearby, GET /api/satellite/training-report

Thresholds (env-configurable, disclosed): ALERT_THRESHOLD=0.80 (alert only on trained
FIRE/SMOKE above it), UNCERTAIN_THRESHOLD=0.45 (below -> UNCERTAIN), MAX_UPLOAD_MB=10.

Example JSON response (values from live inference; null/unavailable when unsupported):
```json
{
  "prediction": "SMOKE",
  "confidence": 0.8052,
  "probabilities": {"fire": 0.02, "smoke": 0.8052, "no_fire": 0.1748},
  "severity": {"level": "HIGH", "score": 0.6055, "method": "0.40*confidence + 0.40*event_likelihood + 0.20*area/25"},
  "risk": {"level": "HIGH", "score": 0.9446, "method": "risk = 1 - P(no_fire)"},
  "growth": {"status": "INSUFFICIENT_TEMPORAL_DATA", "probability": null,
             "message": "A reliable temporal forecast requires suitable sequential observations."},
  "location": null,
  "affected_area": null,
  "model_status": "trained",
  "evidence": ["sklearn-mlp softmax probabilities: {...}"]
}

Example JSON response:
```json
{
  "detection": "smoke",
  "confidence": 0.93,
  "confidence_percentage": 93.0,
  "bounding_boxes": [],
  "severity": "HIGH",
  "growth": {"probability": null, "level": "INSUFFICIENT_DATA"},
  "latitude": null,
  "longitude": null,
  "location_status": "Location unavailable",
  "affected_area": 8.4,
  "area_unit": "image-space-percent",
  "timestamp": "2026-10-08T12:00:00",
  "alert": true,
  "alert_status": "HIGH-RISK WILDFIRE DETECTED"
}
```

## 18. Screenshots Section
Add screenshots here after running the demo dashboard.

## 19. Limitations
- Public NASA datasets are large and require download authorization in some cases.
- Real wildfire detection models need substantial labeled data.
- A demo system can work without a trained model but cannot claim production-level accuracy.
- The app uses SQLite by default and is easily extensible to PostgreSQL or MySQL.

## 20. Future Enhancements
- Integrate real NASA API feeds.
- Add a trained YOLO model checkpoint.
- Add geospatial risk layers.
- Add user authentication.
- Deploy to Azure/Cloud.

## 21. Conclusion
This project provides a practical college-level wildfire monitoring system with a working demo, modular AI pipeline, alerting, and dashboard. The primary implementation follows YOLO + EfficientNet + LSTM for real detection and risk prediction, while other hybrid models remain optional for research and comparison.

## 22. Important notes for the dataset
The project uses NASA Earthdata references, especially MODIS and VIIRS. The system is designed so the dataset can be placed in:

```text
forest_fire_ai/dataset/train/
forest_fire_ai/dataset/val/
forest_fire_ai/dataset/test/
```

If the downloaded NASA data uses different labels, the preprocessing converter maps them into the required classes:
- fire
- smoke
- no_fire
- cloud
- fog
- dust

## 23. Demo mode
The system runs in demo mode without a trained model. All demo outputs are clearly labeled and must not be mistaken for real accuracy metrics.

---

### Project structure summary
```text
forest_fire_ai/
  backend/
  frontend/
  ai/
  dataset/
  outputs/
  scripts/
  requirements.txt
  README.md
  .env.example
```
