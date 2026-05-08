"""
gateway/routes/identification.py
Identification and enrollment endpoints, plus model training triggers.
"""
from collections import Counter, defaultdict
from io import BytesIO
import time
import numpy as np
import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from database.crud import (
    UserCRUD, FeatureProfileCRUD, IdentificationLogCRUD, ModelCRUD,
)
from database.schemas import IdentificationLog, TrainedModelRecord
from services.identification.predictor import Predictor
from services.identification.trainer import ModelTrainer

log = structlog.get_logger()

router = APIRouter(prefix="/api", tags=["Identification"])

# Shared instances (initialized in gateway main.py startup)
predictor: Optional[Predictor] = None
trainer: Optional[ModelTrainer] = None


def init_predictor(p: Predictor, t: ModelTrainer):
    """Called from gateway main to set shared instances."""
    global predictor, trainer
    predictor = p
    trainer = t


async def _build_user_name_map() -> Dict[str, str]:
    users = await UserCRUD.list_all()
    return {u["user_id"]: u["name"] for u in users}


def _normalize_identified_name(raw_value: Optional[str], user_name_map: Dict[str, str]) -> str:
    if not raw_value:
        return "Unknown Person"
    return user_name_map.get(raw_value, raw_value if raw_value != "unknown" else "Unknown Person")


# ── Schemas ───────────────────────────────────────────────────────────────────

class IdentifyRequest(BaseModel):
    static_features: Optional[List[float]] = None
    gait_sequence: Optional[List[List[float]]] = None


class EnrollFrameRequest(BaseModel):
    user_id: str
    static_features: List[float]
    gait_features: Optional[List[float]] = None


class TrainRequest(BaseModel):
    model_type: str = "ensemble"  # svm | lstm | ensemble
    epochs: int = 100
    batch_size: int = 32


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/identify")
async def identify(req: IdentifyRequest):
    """Identify a person from extracted features."""
    if predictor is None or not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    static = np.array(req.static_features) if req.static_features else None
    gait = np.array(req.gait_sequence) if req.gait_sequence else None

    result = predictor.identify(static_features=static, gait_sequence=gait)

    # Log the identification attempt
    log_entry = IdentificationLog(
        predicted_user_id=result.get("predicted_user"),
        confidence=result.get("confidence", 0),
        svm_confidence=result.get("svm_prediction", {}).get("confidence", 0) if result.get("svm_prediction") else 0,
        lstm_confidence=result.get("lstm_prediction", {}).get("confidence", 0) if result.get("lstm_prediction") else 0,
        feature_vector=req.static_features or [],
        model_version=result.get("method", ""),
        latency_ms=result.get("latency_ms", 0),
    )
    await IdentificationLogCRUD.log_identification(log_entry)

    return result


@router.post("/enroll/frame")
async def enroll_frame(req: EnrollFrameRequest):
    """Add a single enrollment frame's features for a user."""
    from config import settings

    user = await UserCRUD.get_by_id(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Store features
    await FeatureProfileCRUD.upsert(
        user_id=req.user_id,
        static_vector=req.static_features,
        gait_sequence=[req.gait_features] if req.gait_features else None,
    )

    # Update enrollment progress
    profile = await FeatureProfileCRUD.get_by_user(req.user_id)
    count = profile["sample_count"] if profile else 0
    min_frames = settings.min_enrollment_frames  # configurable (default 50)
    status = "completed" if count >= min_frames else "in_progress"

    await UserCRUD.update_enrollment_status(req.user_id, status, count)

    return {
        "user_id": req.user_id,
        "frames_collected": count,
        "status": status,
        "progress": min(count / min_frames * 100, 100),
    }


@router.post("/train")
async def train_model(req: TrainRequest):
    """Trigger model training using all enrolled user data."""
    if trainer is None:
        raise HTTPException(status_code=500, detail="Trainer not initialized")

    # Get training data from MongoDB
    data = await FeatureProfileCRUD.get_training_data()

    static_X = data["static_X"]
    static_y = data["static_y"]
    gait_X = data["gait_X"]
    gait_y = data["gait_y"]

    if len(static_X) == 0:
        raise HTTPException(status_code=400, detail="No training data available. Enroll users first.")

    log.info(
        "training_triggered",
        model_type=req.model_type,
        static_samples=len(static_X),
        gait_samples=len(gait_X),
    )

    t_start = time.perf_counter()

    if req.model_type in ("svm", "ensemble"):
        svm_result = await trainer.train_svm(static_X, static_y)
        if svm_result["success"]:
            record = TrainedModelRecord(
                model_type="svm",
                version=svm_result["version"],
                num_classes=svm_result["metrics"]["num_classes"],
                accuracy=svm_result["metrics"]["train_accuracy"],
                f1_score=svm_result["metrics"]["train_f1_macro"],
                model_path="./models",
                is_active=True,
                metrics=svm_result["metrics"],
            )
            await ModelCRUD.save_record(record)
    else:
        svm_result = None

    lstm_result = None
    if req.model_type in ("lstm", "ensemble") and len(gait_X) > 0:
        # Reshape gait data for LSTM: (n_samples, seq_len, n_features)
        # Each gait sample is already a sequence stored as a flat array
        # We need to reshape based on the LSTM expected input
        try:
            # If gait_X is 2D (flat gait features), it's per-frame gait stats, not sequences
            # For LSTM we need sequences — try to reconstruct from raw angle data
            lstm_result = await trainer.train_lstm(
                gait_X.reshape(-1, 30, 8) if gait_X.ndim == 2 and gait_X.shape[1] == 240 else gait_X,
                gait_y,
                epochs=req.epochs,
                batch_size=req.batch_size,
            )
            if lstm_result and lstm_result.get("success"):
                record = TrainedModelRecord(
                    model_type="lstm",
                    version=lstm_result["version"],
                    num_classes=lstm_result["metrics"]["num_classes"],
                    accuracy=lstm_result["metrics"].get("val_accuracy", 0),
                    f1_score=lstm_result["metrics"].get("best_val_f1", 0),
                    model_path="./models",
                    is_active=True,
                    metrics=lstm_result["metrics"],
                )
                await ModelCRUD.save_record(record)
        except Exception as e:
            log.warning("lstm_training_skipped", error=str(e))
            lstm_result = {"success": False, "message": str(e)}

    # Reload models
    if predictor is not None:
        predictor.load_models()

    duration = (time.perf_counter() - t_start) * 1000

    return {
        "success": True,
        "duration_ms": round(duration, 2),
        "svm": svm_result,
        "lstm": lstm_result,
    }


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    user_count = await UserCRUD.count()
    id_stats = await IdentificationLogCRUD.get_stats()
    recent = await IdentificationLogCRUD.get_recent(limit=10)
    models = await ModelCRUD.list_all()

    return {
        "total_users": user_count,
        "identification_stats": id_stats,
        "recent_identifications": recent,
        "trained_models": models,
        "model_status": {
            "svm_ready": predictor.ensemble.svm_ready if predictor else False,
            "lstm_ready": predictor.ensemble.lstm_ready if predictor else False,
        },
    }


@router.get("/models")
async def list_models():
    """List all trained model records."""
    return await ModelCRUD.list_all()


@router.get("/report/pdf")
async def download_pdf_report():
    """Generate a PDF report of identified people and their confidence levels."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    logs = await IdentificationLogCRUD.get_all()
    user_name_map = await _build_user_name_map()

    today = datetime.utcnow().date()
    todays_logs = []
    for item in logs:
        ts = item.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                continue
        if ts and ts.date() == today:
            todays_logs.append(item)

    if not todays_logs:
        todays_logs = logs

    total_scans = len(todays_logs)
    known_rows = []
    unknown_count = 0
    confidence_values = []
    person_counts = Counter()
    person_confidences = defaultdict(list)

    for log_item in todays_logs:
        raw_name = log_item.get("predicted_user_id")
        confidence = float(log_item.get("confidence", 0) or 0)
        display_name = _normalize_identified_name(raw_name, user_name_map)
        confidence_values.append(confidence)

        if display_name == "Unknown Person":
            unknown_count += 1
        else:
            person_counts[display_name] += 1
            person_confidences[display_name].append(confidence)

        known_rows.append(
            [
                str(log_item.get("timestamp", ""))[:19].replace("T", " "),
                display_name,
                f"{round(confidence * 100, 1)}%",
                f"{round(float(log_item.get('latency_ms', 0) or 0), 2)} ms",
                log_item.get("model_version", "—") or "—",
            ]
        )

    average_confidence = (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.45 * inch,
        title="Daily Identification Report",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=22, leading=26, textColor=colors.HexColor("#111827")))
    styles.add(ParagraphStyle(name="ReportSub", parent=styles["Normal"], fontSize=10, leading=12, textColor=colors.HexColor("#4b5563")))

    story = []
    story.append(Paragraph("Daily Identification Report", styles["ReportTitle"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["ReportSub"]))
    story.append(Paragraph(f"Total scans: {total_scans} | Unique identified people: {len(person_counts)} | Unknown scans: {unknown_count}", styles["ReportSub"]))
    story.append(Paragraph(f"Average confidence: {round(average_confidence * 100, 1)}%", styles["ReportSub"]))
    story.append(Spacer(1, 0.18 * inch))

    summary_data = [["Person", "Appearances", "Avg Confidence"]]
    for person, count in person_counts.most_common():
        avg_conf = sum(person_confidences[person]) / len(person_confidences[person]) if person_confidences[person] else 0.0
        summary_data.append([person, str(count), f"{round(avg_conf * 100, 1)}%"])

    if unknown_count > 0:
        summary_data.append(["Unknown Person", str(unknown_count), "0%"])

    if len(summary_data) == 1:
        summary_data.append(["No identified people", "0", "0%"])

    summary_table = Table(summary_data, colWidths=[3.0 * inch, 1.2 * inch, 1.5 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef2ff")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(Paragraph("People Summary", styles["Heading2"]))
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    detail_data = [["Time", "Person", "Confidence", "Latency", "Method"]] + known_rows
    detail_table = Table(detail_data, colWidths=[1.8 * inch, 2.2 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch], repeatRows=1)
    detail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(Paragraph("Detection Log", styles["Heading2"]))
    story.append(detail_table)

    def _add_page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(doc_obj.pagesize[0] - 0.45 * inch, 0.3 * inch, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    buffer.seek(0)

    filename = f"identification-report-{today.isoformat()}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buffer, media_type="application/pdf", headers=headers)
