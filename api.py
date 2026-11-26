"""
api.py
======
FastAPI REST Inference Backend for ResQ_Ai Phase 2.

Exposes:
- POST /predict: Multi-branch visual prediction (Swin-B + Stacked CNN + Handcrafted Descriptors)
                 with Explainable AI (XAI) biomarkers.
- GET /health: Health check, loaded models status, and taxonomy metadata.
- GET /classes: List of the 8 insect bite diagnostic classes.
"""

import os
import sys
import io
import json
import pathlib
import logging
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

BASE_DIR = pathlib.Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.meta_learner import TriBranchMetaLearner, CLASSES, DEFAULT_BRANCH_WEIGHTS
from src.triage.image_question_engine import VisualTriageQuestionEngine, QUESTION_BANK

logger = logging.getLogger("FastAPIBackend")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="ResQ_Ai Multimodal Insect Bite API",
    description="Tri-Branch Diagnostic Vision & Clinical Feature Fusion System",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model holder
orchestrator: Optional[TriBranchMetaLearner] = None
question_engine: Optional[VisualTriageQuestionEngine] = None


def get_orchestrator() -> TriBranchMetaLearner:
    global orchestrator
    if orchestrator is None:
        logger.info("Initializing TriBranchMetaLearner...")
        orchestrator = TriBranchMetaLearner()
    return orchestrator


def get_question_engine() -> VisualTriageQuestionEngine:
    global question_engine
    if question_engine is None:
        logger.info("Initializing VisualTriageQuestionEngine...")
        question_engine = VisualTriageQuestionEngine()
    return question_engine


@app.on_event("startup")
def startup_event():
    logger.info("Triggering TriBranchMetaLearner startup initialization...")
    get_orchestrator()
    get_question_engine()
    logger.info("TriBranchMetaLearner and VisualTriageQuestionEngine ready for requests.")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health status and loaded architecture verification."""
    orch = get_orchestrator()
    q_engine = get_question_engine()
    return {
        "status": "healthy",
        "service": "ResQ_Ai Multimodal Insect Bite Classifier",
        "version": "2.0.0",
        "models": {
            "pytorch_swin_b": orch.swin_classifier is not None,
            "keras_stacked_cnn": orch.keras_model is not None,
            "descriptor_classifier": orch.descriptor_classifier.is_loaded(),
            "visual_triage_question_engine": q_engine is not None,
        },
        "classes": CLASSES,
    }


@app.get("/classes", status_code=status.HTTP_200_OK)
def get_classes():
    """Returns the 8 target insect bite classes."""
    return {"classes": CLASSES, "total": len(CLASSES)}


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_bite(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG, WEBP)"),
    mode: str = Form("tri_branch", description="Inference mode: 'tri_branch', 'ensemble', 'swin', 'keras', or 'descriptor'"),
):
    """
    Evaluates an uploaded insect bite image across the vision branches.

    Returns:
        JSON response with predicted class, confidence, multi-branch probabilities,
        and interpretable XAI clinical biomarkers (erythema, necrosis, circularity).
    """
    global orchestrator
    if orchestrator is None:
        orchestrator = TriBranchMetaLearner()

    # Validate image MIME
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file type '{file.content_type}' is not a valid image.",
        )

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {str(e)}",
        )

    try:
        result = orchestrator.predict(pil_img)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )

    # Convert branch probabilities to dicts
    branch_dict = {
        branch: {cls_name: float(prob) for cls_name, prob in zip(CLASSES, probs)}
        for branch, probs in result["branch_probabilities"].items()
    }

    response_payload = {
        "filename": file.filename,
        "mode": mode,
        "predicted_class": result["predicted_class"],
        "confidence": round(result["confidence"], 2),
        "fused_probabilities": {
            cls_name: round(float(prob), 4)
            for cls_name, prob in result["fused_probabilities_dict"].items()
        },
        "branch_probabilities": branch_dict,
        "visual_metrics_xai": result["visual_metrics"],
    }
    return JSONResponse(content=response_payload)


@app.post("/predict/interactive", status_code=status.HTTP_200_OK)
async def predict_interactive(
    file: UploadFile = File(..., description="Target image file (JPEG, PNG, WEBP)"),
    mode: str = Form("tri_branch", description="Inference mode: 'tri_branch', 'ensemble', 'swin', 'keras', or 'descriptor'"),
    answered_questions: Optional[str] = Form(None, description="Optional JSON string mapping question_id -> 'Yes'/'No'/'Not sure'"),
    max_questions: int = Form(2, description="Maximum number of clarification questions to return"),
):
    """
    Interactive Visual-Triage Inference:
    Evaluates an uploaded insect bite image, measures prediction uncertainty,
    recommends high-information discriminative questions conditioned on visual biomarkers,
    and updates posterior probabilities dynamically based on patient responses.
    """
    orch = get_orchestrator()
    q_engine = get_question_engine()

    # Validate image MIME
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file type '{file.content_type}' is not a valid image.",
        )

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {str(e)}",
        )

    try:
        result = orch.predict(pil_img)
    except Exception as e:
        logger.error(f"Interactive prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )

    initial_probs = np.array(result["fused_probabilities"], dtype=np.float32)
    visual_metrics = result["visual_metrics"]

    # Parse answered questions if supplied
    parsed_answers: Dict[str, str] = {}
    if answered_questions:
        try:
            parsed_answers = json.loads(answered_questions)
            if not isinstance(parsed_answers, dict):
                raise ValueError("answered_questions JSON must be an object/dict")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid answered_questions JSON format: {str(e)}",
            )

    # Sequentially update probabilities if answers exist
    if parsed_answers:
        current_probs = q_engine.process_answered_questions(initial_probs, parsed_answers)
    else:
        current_probs = initial_probs.copy()

    # Evaluate clarification need & rank questions
    asked_list = list(parsed_answers.keys())
    clarification_res = q_engine.select_clarification_questions(
        current_probs,
        visual_descriptors=visual_metrics,
        asked_questions=asked_list,
        max_questions=max_questions,
    )

    top_idx = int(np.argmax(current_probs))
    top_class = CLASSES[top_idx]
    top_conf = float(current_probs[top_idx] * 100.0)

    response_payload = {
        "filename": file.filename,
        "mode": mode,
        "predicted_class": top_class,
        "confidence": round(top_conf, 2),
        "initial_probabilities": {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(CLASSES, initial_probs)
        },
        "current_probabilities": {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(CLASSES, current_probs)
        },
        "needs_clarification": clarification_res["needs_clarification"],
        "uncertainty_metrics": clarification_res["uncertainty_metrics"],
        "questions": clarification_res["questions"],
        "clarification_explanation": clarification_res["explanation"],
        "answered_questions": parsed_answers,
        "visual_metrics_xai": visual_metrics,
    }
    return JSONResponse(content=response_payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
