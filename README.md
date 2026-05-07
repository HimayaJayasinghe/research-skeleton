# 🦴 Skeleton-Based Person Identification System

Real-time person identification using **skeletal bone structure** and **gait patterns** — not face recognition.

## Architecture

```
   Webcam → Video Processing → Pose Estimation (MediaPipe)
       → Feature Extraction (42 static + 15 gait features)
       → Identification (SVM + LSTM Ensemble)
       → Dashboard / API Response
```

**Tech Stack**: Python, FastAPI, MediaPipe, OpenCV, PyTorch, scikit-learn, MongoDB

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MongoDB

```bash
# Using Docker
docker run -d --name mongodb -p 27017:27017 mongo:7

# Or use your existing MongoDB installation
```

### 3. Run the System

```bash
python run_all_services.py
```

Open **http://localhost:8000** for the web dashboard.

---

## Usage

### CLI Workflow

```bash
# Step 1: Enroll users (capture 60s of skeleton data per person)
python scripts/enroll_user.py --name "Alice" --duration 60
python scripts/enroll_user.py --name "Bob" --duration 60

# Step 2: Train models (SVM + LSTM)
python scripts/train_model.py --type ensemble --epochs 100

# Step 3: Run real-time identification
python scripts/run_identification.py
```

### Web Dashboard

1. Start system: `python run_all_services.py`
2. Open **http://localhost:8000**
3. Use the **Enroll** tab to register users via webcam
4. Use the **Training** tab to train models
5. Use the **Live Feed** tab for real-time identification

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health check |
| `WS` | `/ws/stream` | Real-time video processing (WebSocket) |
| `GET` | `/api/users/` | List all users |
| `POST` | `/api/users/` | Create new user |
| `DELETE` | `/api/users/{id}` | Delete user |
| `POST` | `/api/enroll/frame` | Add enrollment frame |
| `POST` | `/api/identify` | Identify from features |
| `POST` | `/api/train` | Train models |
| `GET` | `/api/stats` | System statistics |

Full API docs at: **http://localhost:8000/docs**

---

## Model Architecture

### Static Features → SVM
- 42-dimensional feature vector (limb lengths, joint angles, body proportions)
- Scale-invariant via torso-length normalization
- SVM with RBF kernel, probability output

### Sequence Features → LSTM
- 30-frame sliding window of 8 joint angles (30×8 matrix)
- Bidirectional LSTM → FC classifier
- Captures gait cycle patterns unique to individuals

### Ensemble Fusion
```
final_score = svm_weight × svm_confidence + lstm_weight × lstm_confidence
```

---

## Project Structure

```
skeletal-Identification/
├── gateway/                    # API Gateway (FastAPI)
│   ├── main.py                 # Entry point
│   └── routes/                 # REST & WebSocket endpoints
├── services/
│   ├── video_processing/       # Frame capture & preprocessing
│   ├── pose_estimation/        # MediaPipe skeleton extraction
│   ├── feature_extraction/     # Static + gait features
│   └── identification/         # SVM + LSTM + Ensemble
├── database/                   # MongoDB connection, schemas, CRUD
├── dashboard/                  # Web UI (HTML/CSS/JS)
├── scripts/                    # CLI tools (enroll, train, identify)
├── models/                     # Saved trained models
├── config.py                   # Centralized settings
├── run_all_services.py         # Main launcher
└── requirements.txt
```

---

## Configuration

Edit `.env` to customize:

```env
MONGODB_URI=mongodb://localhost:27017
CONFIDENCE_THRESHOLD=0.65
SVM_WEIGHT=0.5
LSTM_WEIGHT=0.5
MEDIAPIPE_MODEL_COMPLEXITY=1
LSTM_SEQUENCE_LENGTH=30
LSTM_EPOCHS=100
```
