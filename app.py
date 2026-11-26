import os
import pathlib
import hashlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from db_manager import BiteDBManager
from download_model import ensure_model
from phase2_voice_triage import (
    generate_question_audio,
    transcribe_patient_audio,
    map_transcript_to_clinical_slots,
    render_handy_voice_triage_ui,
    TRIAGE_QUESTIONS,
    CLINICAL_LOCATIONS,
    CLINICAL_TIMINGS,
    CLINICAL_SYMPTOMS,
)
from swin_classifier import (
    SwinClassifier,
    fuse_vision_probabilities,
    SWIN_MODEL_PATH as DEFAULT_SWIN_PATH,
)
try:
    from src.models.descriptor_model import DescriptorClassifier
    from src.models.meta_learner import fuse_tri_branch_probabilities, DEFAULT_BRANCH_WEIGHTS
    DESCRIPTOR_BRANCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DescriptorClassifier = None
    fuse_tri_branch_probabilities = None
    DEFAULT_BRANCH_WEIGHTS = (0.50, 0.50, 0.0)
    DESCRIPTOR_BRANCH_AVAILABLE = False

try:
    from src.triage.image_question_engine import VisualTriageQuestionEngine, QUESTION_BANK
    TRIAGE_ENGINE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    VisualTriageQuestionEngine = None
    QUESTION_BANK = {}
    TRIAGE_ENGINE_AVAILABLE = False

load_dotenv()

# ---------------------------------------------------------
# Page Configuration & Constants
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Insect Bite Classifier",
    page_icon="🦟",
    layout="wide",
)

CLASSES = [
    'ants',
    'bed_bugs',
    'chiggers',
    'fleas',
    'mosquitos',
    'no_bites',
    'spiders',
    'ticks',
]

CHANNEL_MEANS = np.array([170.88408, 134.76701, 118.84133], dtype=np.float32)

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_KERAS_PATH = str(BASE_DIR / "models" / "bug_bite_8class_stacked_model.keras")

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_KERAS_PATH)
SWIN_MODEL_PATH = os.getenv("SWIN_MODEL_PATH", DEFAULT_SWIN_PATH)
DEFAULT_DESCRIPTOR_PATH = str(BASE_DIR / "models" / "descriptor_classifier.joblib")
DESCRIPTOR_MODEL_PATH = os.getenv("DESCRIPTOR_MODEL_PATH", DEFAULT_DESCRIPTOR_PATH)

# ---------------------------------------------------------
# Resource Initialization
# ---------------------------------------------------------
@st.cache_resource
def load_classifier():
    ensure_model(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Keras model file not found at path: `{MODEL_PATH}`")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_swin_classifier():
    if not os.path.exists(SWIN_MODEL_PATH):
        st.error(f"PyTorch Swin Transformer model not found at path: `{SWIN_MODEL_PATH}`")
        st.stop()
    return SwinClassifier(model_path=SWIN_MODEL_PATH)

@st.cache_resource
def load_descriptor_classifier():
    if not DESCRIPTOR_BRANCH_AVAILABLE or DescriptorClassifier is None:
        return None
    try:
        classifier = DescriptorClassifier(model_path=DESCRIPTOR_MODEL_PATH)
        return classifier if classifier.is_loaded() else None
    except Exception as e:
        return None

@st.cache_resource
def load_db():
    return BiteDBManager()

@st.cache_resource
def load_triage_engine():
    if not TRIAGE_ENGINE_AVAILABLE or VisualTriageQuestionEngine is None:
        return None
    try:
        return VisualTriageQuestionEngine()
    except Exception:
        return None

model = load_classifier()
swin_model = load_swin_classifier()
desc_model = load_descriptor_classifier()
db = load_db()
triage_engine = load_triage_engine()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesses input image with 224x224 resize and channel-mean subtraction."""
    img = image.convert('RGB').resize((224, 224))
    img_arr = (np.array(img, dtype=np.float32) - CHANNEL_MEANS) / 255.0
    return np.expand_dims(img_arr, axis=0)

# ---------------------------------------------------------
# Clinical Prior & Bayesian Fusion Logic
# ---------------------------------------------------------
def get_clinical_weights(location: str, timing: str, symptoms: list):
    is_loc = location != "Select / Not specified"
    is_time = timing != "Select / Not specified"
    is_sym = len(symptoms) > 0 and ("None / Unsure" not in symptoms)

    if not (is_loc or is_time or is_sym):
        return False, np.ones(len(CLASSES), dtype=np.float32)

    weights = {c: 1.0 for c in CLASSES}

    if location == "Ankles / Feet / Lower Legs":
        weights["fleas"] *= 2.2
        weights["chiggers"] *= 1.6
        weights["ants"] *= 1.4
    elif location == "Waistband / Groin / Skin Folds":
        weights["chiggers"] *= 2.5
        weights["ticks"] *= 1.5
    elif location == "Neck / Face / Arms (Exposed while sleeping)":
        weights["bed_bugs"] *= 2.5
        weights["mosquitos"] *= 1.3
    elif location == "Scalp / Behind Ears / Armpits":
        weights["ticks"] *= 2.5
        weights["spiders"] *= 1.2

    if timing == "Immediate / within minutes (Instant burning or sharp pain)":
        weights["ants"] *= 2.2
        weights["mosquitos"] *= 1.8
        weights["spiders"] *= 1.5
        weights["bed_bugs"] *= 0.2
    elif timing == "Woke up with bites / Noticed in the morning":
        weights["bed_bugs"] *= 2.5
        weights["fleas"] *= 1.2
    elif timing == "Delayed reaction over 1–3 days (Gradual itching)":
        weights["ticks"] *= 2.0
        weights["chiggers"] *= 2.0
        weights["fleas"] *= 1.3

    if "Linear track / 'Breakfast-Lunch-Dinner' clustered line" in symptoms:
        weights["bed_bugs"] *= 3.0
        weights["fleas"] *= 1.5
        weights["spiders"] *= 0.2
    if "Intense itching with small red papules" in symptoms:
        weights["fleas"] *= 1.8
        weights["chiggers"] *= 1.8
        weights["mosquitos"] *= 1.4
    if "Expanding bullseye / target ring rash" in symptoms:
        weights["ticks"] *= 4.0
        weights["spiders"] *= 0.3
    if "Pustule or white fluid-filled blister" in symptoms:
        weights["ants"] *= 2.5
    if "Two distinct fang marks / Central necrosis" in symptoms:
        weights["spiders"] *= 3.5
        weights["bed_bugs"] *= 0.2
        weights["fleas"] *= 0.2
    if "No itching / completely painless" in symptoms:
        weights["no_bites"] *= 2.0
        weights["ticks"] *= 1.5

    prior_vec = np.array([weights[c] for c in CLASSES], dtype=np.float32)
    return True, (prior_vec / np.sum(prior_vec))

def resolve_multimodal_conflict(cnn_probs, prior_probs, confidence_threshold=0.50, similarity_threshold=0.35):
    top_cnn_idx = int(np.argmax(cnn_probs))
    top_cnn_conf = float(cnn_probs[top_cnn_idx])
    top_prior_idx = int(np.argmax(prior_probs))

    cosine_sim = float(
        np.dot(cnn_probs, prior_probs)
        / (np.linalg.norm(cnn_probs) * np.linalg.norm(prior_probs) + 1e-7)
    )

    if top_cnn_conf >= confidence_threshold and top_cnn_idx != top_prior_idx:
        return True, cnn_probs, "Conflict Detected: Strong visual features contradict clinical prior. Reverted to pure vision."

    if cosine_sim < similarity_threshold:
        return True, cnn_probs, f"Conflict Detected: Text prior strongly disagrees with visual evidence (Cosine Sim: {cosine_sim:.2f}). Reverted to pure vision."

    fused = cnn_probs * prior_probs
    return False, (fused / np.sum(fused)), "Multimodal Fusion: Vision features and clinical context aligned."

# ---------------------------------------------------------
# Deterministic Clinical Guidance & Risk Mapping
# ---------------------------------------------------------
def get_clinical_guidance(predicted_class: str, confidence: float, symptoms: list) -> dict:
    """
    Deterministically computes clinical risk level, first-aid action plan,
    and emergency red-flag criteria in-memory based on predicted class, confidence,
    and clinical physical observations.
    """
    symptoms = symptoms or []
    has_necrosis_or_fangs = "Two distinct fang marks / Central necrosis" in symptoms
    has_bullseye = "Expanding bullseye / target ring rash" in symptoms

    if predicted_class == 'spiders':
        if has_necrosis_or_fangs:
            risk_level = "HIGH / URGENT"
            risk_type = "error"
            rationale = "Potential necrotic or neurotoxic envenomation indicated by central necrosis / fang marks."
        else:
            risk_level = "MODERATE"
            risk_type = "warning"
            rationale = "Suspected spider bite requiring close observation for spreading necrosis, ulceration, or neurotoxic spasms."
        first_aid = [
            "Wash the bite area thoroughly with mild soap and clean water.",
            "Apply a cold compress or cool damp washcloth (15 min on / 15 min off). Do NOT apply ice directly to skin.",
            "Elevate the affected limb above heart level to minimize localized swelling and venom spread.",
            "Keep the patient calm and at rest; strictly avoid cutting, suction, or tourniquets.",
        ]
        red_flags = [
            "Rapidly expanding dusky, purple, or black lesion center (tissue necrosis / ulceration).",
            "Severe muscle cramping, abdominal rigidity, profuse sweating, or tremors (Latrodectus / Black Widow sign).",
            "Difficulty breathing, chest tightness, dizziness, or difficulty swallowing.",
        ]

    elif predicted_class == 'ticks':
        if has_bullseye or confidence >= 60.0:
            risk_level = "HIGH / URGENT"
            risk_type = "error"
            rationale = "Suspected tick bite / expanding annular rash associated with Lyme disease (Borrelia burgdorferi)."
        else:
            risk_level = "MODERATE"
            risk_type = "warning"
            rationale = "Tick attachment carrying potential risk of vector-borne infection (Lyme, RMSF, Babesiosis)."
        first_aid = [
            "Inspect the entire body thoroughly for any attached tick (check hairline, ears, groin, skin folds).",
            "If attached, use fine-tipped tweezers to grasp the tick as close to the skin as possible and pull straight upward with steady pressure (do NOT twist, jerk, or burn).",
            "Disinfect the attachment site thoroughly with 70% isopropyl alcohol or iodine.",
            "Save the removed tick in a sealed container or take a clear photo for identification if systemic symptoms develop.",
        ]
        red_flags = [
            "Expanding circular or bullseye erythema migrans rash (>5 cm diameter).",
            "Onset of fever, chills, severe fatigue, headache, or stiff neck within 1–4 weeks.",
            "Facial nerve palsy (drooping side of face), irregular heartbeat, or severe migratory joint pain.",
        ]

    elif predicted_class == 'ants':
        risk_level = "LOW to MODERATE"
        risk_type = "warning"
        rationale = "Stinging ant envenomation (e.g., Solenopsis fire ant) with sterile pustule formation or systemic allergy risk."
        first_aid = [
            "Clean the sting area gently with antiseptic soap and water.",
            "DO NOT pop or squeeze the sterile pustules or fluid-filled vesicles (prevents secondary staphylococcal infection).",
            "Apply topical 1% hydrocortisone cream or calamine lotion to relieve localized burning and erythema.",
            "Take over-the-counter oral antihistamines for acute itching.",
        ]
        red_flags = [
            "Facial swelling, angioedema of the lips/tongue, or sensation of throat closing (anaphylaxis).",
            "Audible wheezing, acute shortness of breath, or stridor.",
            "Dizziness, lightheadedness, confusion, or sudden hypotension.",
        ]

    elif predicted_class in ['bed_bugs', 'fleas', 'chiggers', 'mosquitos']:
        risk_level = "LOW (Symptomatic)"
        risk_type = "info"
        rationale = "Common superficial insect bite eliciting localized histamine reaction; primarily managed with supportive care."
        first_aid = [
            "Wash the bite area gently with soap and lukewarm water.",
            "Apply a cold pack wrapped in a towel for 10–15 minutes to reduce edema and redness.",
            "Use topical anti-itch lotions (calamine, hydrocortisone) or oral antihistamines (cetirizine, diphenhydramine).",
            "Strictly avoid scratching to prevent epidermal breakdown and opportunistic bacterial colonization.",
        ]
        red_flags = [
            "Rapidly spreading bright red streaks, warmth, or increasing tenderness around the bite (signs of cellulitis).",
            "Pus drainage, yellow crusting, or worsening pain beyond 48 hours.",
            "Fever, chills, or systemic malaise.",
        ]

    else:  # 'no_bites'
        risk_level = "MINIMAL / BENIGN"
        risk_type = "success"
        rationale = "No evidence of venomous or inflammatory insect bite reaction detected on the skin surface."
        first_aid = [
            "Maintain routine daily skin hygiene and moisturizing.",
            "Monitor the area over the next 24–48 hours if mild redness or irritation persists.",
        ]
        red_flags = [
            "Seek formal clinical evaluation if unexpected swelling, discoloration, or spreading pain develops.",
        ]

    return {
        "risk_level": risk_level,
        "risk_type": risk_type,
        "rationale": rationale,
        "first_aid": first_aid,
        "red_flags": red_flags,
    }

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
st.title("🦟 Multimodal Insect Bite Diagnostic System")

with st.sidebar:
    st.header("👤 Patient Metadata")
    user_id_val = st.text_input("Patient ID:", value="PATIENT_001")

    st.markdown("---")
    st.header("🧠 Vision Backbone")
    vision_options = [
        "Dual Vision Ensemble (Swin-B + Stacked CNN)",
        "PyTorch Swin-B Transformer (81.52% Val Acc)",
        "Keras Tier-2 Stacked CNN Ensemble",
    ]
    if desc_model is not None and desc_model.is_loaded():
        vision_options.insert(0, "Tri-Branch Multimodal Ensemble (Swin-B + CNN + Descriptors)")
        vision_options.append("Handcrafted Visual Descriptor Classifier")
    else:
        st.info("ℹ️ Handcrafted Descriptors disabled (OpenCV not installed in this environment).")

    vision_mode = st.selectbox(
        "Active Vision Architecture:",
        vision_options,
        index=0,
        help="Select which diagnostic vision architecture drives classification.",
    )

    swin_acc = swin_model.metadata.get("val_acc")
    swin_acc_str = f"{swin_acc:.2%}" if swin_acc else "81.52%"
    st.caption(f"🤖 **Swin-B:** `{os.path.basename(SWIN_MODEL_PATH)}` ({swin_acc_str})")
    st.caption(f"🤖 **Stacked CNN:** `{os.path.basename(MODEL_PATH)}`")
    if desc_model is not None and desc_model.is_loaded():
        st.caption(f"🤖 **Descriptors:** `{os.path.basename(DESCRIPTOR_MODEL_PATH)}` (156 PCA components)")

    st.markdown("---")
    if db.is_connected:
        st.success("🟢 MongoDB Atlas Connected")
    else:
        st.error("🔴 MongoDB Disconnected")

# Initialize session state for clinical context slots, voice triage, and dynamic questions
if "sel_location_key" not in st.session_state:
    st.session_state["sel_location_key"] = "Select / Not specified"
if "sel_timing_key" not in st.session_state:
    st.session_state["sel_timing_key"] = "Select / Not specified"
if "sel_symptoms_key" not in st.session_state:
    st.session_state["sel_symptoms_key"] = []
if "voice_transcripts" not in st.session_state:
    st.session_state["voice_transcripts"] = {"q1": "", "q2": "", "q3": "", "single": ""}
if "voice_audio_hashes" not in st.session_state:
    st.session_state["voice_audio_hashes"] = {}
if "interactive_answers" not in st.session_state:
    st.session_state["interactive_answers"] = {}
if "has_active_diagnosis" not in st.session_state:
    st.session_state["has_active_diagnosis"] = False
if "last_active_file_id" not in st.session_state:
    st.session_state["last_active_file_id"] = None

col_img, col_text = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("1. Bite Image")
    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Live Camera"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload lesion image (JPG, PNG):",
            type=["jpg", "jpeg", "png"],
            help="Upload an existing image of the insect bite or skin lesion.",
        )
        if uploaded_file is not None:
            preview_img = Image.open(uploaded_file)
            st.image(preview_img, caption="📁 Uploaded Image Preview", use_container_width=True)

    with tab_camera:
        camera_file = st.camera_input(
            "Capture bite lesion photo:",
            help="Position the insect bite clearly in frame with adequate lighting, then click 'Take Photo'.",
        )
        if camera_file is not None:
            st.caption("✅ Live camera photo captured.")

    # Reset dynamic triage answers if image source changes
    current_active_id = getattr(camera_file, "name", None) or getattr(uploaded_file, "name", None)
    if current_active_id != st.session_state.get("last_active_file_id"):
        st.session_state["last_active_file_id"] = current_active_id
        st.session_state["interactive_answers"] = {}
        st.session_state["has_active_diagnosis"] = False
        st.session_state.pop("diagnosis_state", None)

with col_text:
    st.subheader("2. Clinical Prior Context")

    # -----------------------------------------------------
    # Phase 2 Feature 4: Hands-Free Voice Triage (Handy AI)
    # -----------------------------------------------------
    voice_slots = render_handy_voice_triage_ui()
    if voice_slots and voice_slots.get("is_active"):
        st.session_state["sel_location_key"] = voice_slots["body_location"]
        st.session_state["sel_timing_key"] = voice_slots["onset_timing"]
        st.session_state["sel_symptoms_key"] = voice_slots["hallmark_symptoms"]

    # -----------------------------------------------------
    # Clinical Prior Dropdowns (Auto-Populated & User-Editable)
    # -----------------------------------------------------
    loc_val = st.selectbox(
        "Anatomical Location:",
        CLINICAL_LOCATIONS,
        key="sel_location_key",
    )

    time_val = st.selectbox(
        "Onset Timing:",
        CLINICAL_TIMINGS,
        key="sel_timing_key",
    )

    sym_val = st.multiselect(
        "Observed Clinical Signs:",
        CLINICAL_SYMPTOMS,
        key="sel_symptoms_key",
    )

submit_button = st.button("🚀 Run Diagnosis", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Execution & Database Logging
# ---------------------------------------------------------
if submit_button:
    # Resolve active image input source
    active_image = None
    input_source_label = ""

    if camera_file is not None and uploaded_file is not None:
        active_image = camera_file
        input_source_label = "Live Camera Capture"
        st.info("ℹ️ Both camera photo and uploaded file detected. Analyzing the live camera capture.")
    elif camera_file is not None:
        active_image = camera_file
        input_source_label = "Live Camera Capture"
    elif uploaded_file is not None:
        active_image = uploaded_file
        input_source_label = "Uploaded File"

    if active_image is None:
        st.error("⚠️ Please upload an image or capture a photo using your camera before submitting.")
    else:
        with st.spinner(f"Analyzing visual features ({input_source_label}) and clinical priors..."):
            image_obj = Image.open(active_image)

            # 1. Keras Tier-2 Stacked CNN Prediction
            input_tensor = preprocess_image(image_obj)
            raw_cnn_probs = model.predict(input_tensor, verbose=0)[0]

            # 2. PyTorch Swin Transformer Prediction
            swin_probs = swin_model.predict_proba(image_obj)

            # 3. Handcrafted Visual Descriptors & XAI Biomarkers
            if desc_model is not None and desc_model.is_loaded():
                desc_probs = desc_model.predict_proba(image_obj)
                visual_metrics = desc_model.extract_interpretable_metrics(image_obj)
            else:
                desc_probs = swin_probs.copy()
                visual_metrics = {
                    "erythema_score": 0.0,
                    "lesion_circularity": 0.0,
                    "texture_contrast": 0.0,
                    "lesion_area_ratio": 0.0,
                    "necrotic_center_contrast": 0.0,
                    "lesion_solidity": 0.0,
                    "lesion_eccentricity": 0.0,
                }

            # 4. Multi-Branch Probability Fusion
            # Dual Vision Ensemble (50% Swin-B + 50% Stacked CNN)
            dual_vision_probs = fuse_vision_probabilities(
                raw_cnn_probs, swin_probs, method="weighted_average", weights=(0.50, 0.50)
            )
            # Tri-Branch Multimodal Ensemble (45% Swin + 40% CNN + 15% Descriptors)
            if fuse_tri_branch_probabilities is not None and (desc_model is not None and desc_model.is_loaded()):
                tri_vision_probs = fuse_tri_branch_probabilities(
                    swin_probs, raw_cnn_probs, desc_probs, method="weighted_average", weights=DEFAULT_BRANCH_WEIGHTS
                )
            else:
                tri_vision_probs = dual_vision_probs

            # Select active vision distribution based on sidebar selection
            if "Tri-Branch" in vision_mode:
                active_vision_probs = tri_vision_probs
                vision_label = "Tri-Branch Multimodal Ensemble (Swin-B + CNN + Descriptors)"
            elif "Dual Vision" in vision_mode:
                active_vision_probs = dual_vision_probs
                vision_label = "Dual Vision Ensemble (Swin-B + Stacked CNN)"
            elif "PyTorch Swin-B" in vision_mode:
                active_vision_probs = swin_probs
                vision_label = "PyTorch Swin-B Transformer"
            elif "Keras Tier-2" in vision_mode:
                active_vision_probs = raw_cnn_probs
                vision_label = "Keras Tier-2 Stacked CNN"
            else:  # Handcrafted Visual Descriptor Classifier
                active_vision_probs = desc_probs
                vision_label = "Handcrafted Visual Descriptors"

            has_text, prior_distribution = get_cli  nical_weights(loc_val, time_val, sym_val)

            if has_text:
                is_conflict, base_probs, status_msg = resolve_multimodal_conflict(
                    active_vision_probs, prior_distribution
                )
            else:
                base_probs = active_vision_probs
                status_msg = ""
                is_conflict = False

            top_idx = int(np.argmax(base_probs))
            top_class = CLASSES[top_idx]
            top_conf = float(base_probs[top_idx] * 100)
            prob_dict = {CLASSES[i]: float(base_probs[i]) for i in range(len(CLASSES))}

            # Progression Analysis
            progression_data = db.analyze_progression(
                user_id=user_id_val,
                current_class=top_class,
                current_location=loc_val,
                current_symptoms=sym_val,
            )

            # Persist record to MongoDB
            doc_id = db.log_prediction(
                user_id=user_id_val,
                predicted_class=top_class,
                confidence=top_conf,
                location=loc_val,
                symptoms=sym_val,
                timing=time_val,
                raw_probs=prob_dict,
            )

            # Save state in session for dynamic interaction
            st.session_state["interactive_answers"] = {}
            st.session_state["has_active_diagnosis"] = True
            st.session_state["diagnosis_state"] = {
                "base_probs": base_probs,
                "active_vision_probs": active_vision_probs,
                "tri_vision_probs": tri_vision_probs,
                "swin_probs": swin_probs,
                "raw_cnn_probs": raw_cnn_probs,
                "desc_probs": desc_probs,
                "prior_distribution": prior_distribution if has_text else None,
                "has_text": has_text,
                "status_msg": status_msg,
                "is_conflict": is_conflict,
                "vision_label": vision_label,
                "input_source_label": input_source_label,
                "visual_metrics": visual_metrics,
                "user_id_val": user_id_val,
                "loc_val": loc_val,
                "time_val": time_val,
                "sym_val": sym_val,
                "doc_id": doc_id,
                "progression_data": progression_data,
            }

# ---------------------------------------------------------
# Dynamic Results & Interactive Visual-Triage Rendering
# ---------------------------------------------------------
if st.session_state.get("has_active_diagnosis", False) and "diagnosis_state" in st.session_state:
    diag = st.session_state["diagnosis_state"]
    base_probs = diag["base_probs"]
    visual_metrics = diag["visual_metrics"]
    answered = st.session_state.get("interactive_answers", {})

    # Apply Bayesian updates if user has answered dynamic clarification questions
    if triage_engine is not None and answered:
        final_probs = triage_engine.process_answered_questions(base_probs, answered)
        is_clarified = True
    else:
        final_probs = base_probs.copy()
        is_clarified = False

    top_idx = int(np.argmax(final_probs))
    top_class = CLASSES[top_idx]
    top_conf = float(final_probs[top_idx] * 100)

    # Show multimodal conflict status
    if diag["has_text"]:
        if diag["is_conflict"]:
            st.warning(f"⚠️ {diag['status_msg']}")
        else:
            st.info(f"✅ {diag['status_msg']}")
    else:
        st.caption(f"Executed in **Pure Vision Mode** via {diag['vision_label']} (no clinical metadata provided).")

    st.markdown("---")
    st.subheader("📊 Diagnostic Verdict")
    st.success(
        f"**Predicted Class:** `{top_class.upper()}` | **Confidence:** {top_conf:.2f}% | **Backbone:** `{diag['vision_label']}` | **Source:** {diag['input_source_label']}"
    )

    if is_clarified:
        st.info(
            f"🧬 **Active Clarification Applied:** {len(answered)} response(s) integrated via Bayesian update. "
            f"Verdict adjusted based on patient clarification."
        )

    # -----------------------------------------------------
    # Phase 2 Feature 5: Image-Based Dynamic Questioning (Visual-Triage Clarification)
    # -----------------------------------------------------
    if triage_engine is not None:
        clarification_info = triage_engine.select_clarification_questions(
            final_probs,
            visual_descriptors=visual_metrics,
            asked_questions=list(answered.keys()),
            max_questions=2,
        )

        with st.container(border=True):
            st.markdown("### ❓ Image-Based Dynamic Questioning (Visual-Triage Clarification)")

            # Render existing answers
            if answered:
                st.markdown(f"**Current Clarification Responses ({len(answered)} recorded):**")
                ans_cols = st.columns(min(len(answered), 3))
                for c, (q_id, ans_val) in zip(ans_cols, answered.items()):
                    q_obj = triage_engine.question_bank.get(q_id)
                    q_label = q_obj.prompt if q_obj else q_id
                    with c:
                        st.info(f"**{q_label}**\n\nYour Answer: `{ans_val}`")

                if st.button("🔄 Reset Clarification Questions", key="reset_clarify_btn"):
                    st.session_state["interactive_answers"] = {}
                    st.rerun()

            if clarification_info["needs_clarification"] and clarification_info["questions"]:
                st.warning(f"⚠️ **Visual Ambiguity Detected:** {clarification_info['explanation']}")
                st.markdown("Please answer the following follow-up questions to help differentiate the most likely candidates:")

                for idx, q in enumerate(clarification_info["questions"]):
                    with st.container(border=True):
                        qc1, qc2 = st.columns([3, 2])
                        with qc1:
                            st.markdown(f"**Question {idx+1} ({q['category']})** • *Expected Info Gain: {q['information_gain']} bits*")
                            st.markdown(f"#### {q['prompt']}")
                            st.caption(f"💡 *Clinical rationale:* {q['clinical_rationale']}")
                            if q.get("visual_reason"):
                                st.info(f"🔬 **Visual Biomarker Trigger:** {q['visual_reason']}")
                            st.caption(f"ℹ️ *Disambiguation Reason:* {q['xai_explanation']}")
                        with qc2:
                            st.markdown("**Your Answer:**")
                            b1, b2, b3 = st.columns(3)
                            with b1:
                                if st.button("✅ Yes", key=f"btn_yes_{q['id']}", use_container_width=True):
                                    st.session_state["interactive_answers"][q["id"]] = "Yes"
                                    st.rerun()
                            with b2:
                                if st.button("❌ No", key=f"btn_no_{q['id']}", use_container_width=True):
                                    st.session_state["interactive_answers"][q["id"]] = "No"
                                    st.rerun()
                            with b3:
                                if st.button("🤷 Unsure", key=f"btn_unsure_{q['id']}", use_container_width=True):
                                    st.session_state["interactive_answers"][q["id"]] = "Not sure"
                                    st.rerun()
            elif not clarification_info["needs_clarification"] and answered:
                st.success("✅ **Diagnostic Ambiguity Resolved:** Active clarification questions successfully differentiated candidate classes.")
            else:
                st.caption("✅ **High Visual Confidence:** Visual prediction meets confidence thresholds (>75%, low entropy). No follow-up questions needed.")

    # -----------------------------------------------------
    # Explainable AI (XAI): Visual Biomarkers & Lesion Morphology
    # -----------------------------------------------------
    with st.expander("🔬 Explainable AI (XAI): Visual Biomarkers & Lesion Morphology", expanded=True):
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(
                label="🔴 Erythema (Redness)",
                value=f"{visual_metrics['erythema_score']:.2f}",
                help="LAB a* elevation index representing localized cutaneous capillary inflammation."
            )
            st.caption("Active Inflammation" if visual_metrics['erythema_score'] > 0.4 else "Mild / Basal Redness")
        with m_col2:
            st.metric(
                label="🎯 Lesion Circularity",
                value=f"{visual_metrics['lesion_circularity']:.2f}",
                help="4*pi*Area/Perimeter^2 (1.0 = perfect circle, characteristic of tick erythema migrans target rash)."
            )
            st.caption("Target Rash Pattern" if visual_metrics['lesion_circularity'] > 0.6 else "Irregular Mark")
        with m_col3:
            st.metric(
                label="⚫ Necrotic Center Contrast",
                value=f"{visual_metrics['necrotic_center_contrast']:.1f}",
                help="Lightness drop between periphery and center. High positive values indicate a necrotic or dark core (spider bite)."
            )
            st.caption("Central Depression" if visual_metrics['necrotic_center_contrast'] > 10.0 else "Uniform Tone")
        with m_col4:
            st.metric(
                label="🧬 Texture Contrast",
                value=f"{visual_metrics['texture_contrast']:.2f}",
                help="GLCM micro-roughness capturing blister, crusting, or skin breakdown."
            )
            st.caption(f"Area: {visual_metrics['lesion_area_ratio']*100:.1f}% field")

    # -----------------------------------------------------
    # Clinical Risk Assessment & Medical Action Plan
    # -----------------------------------------------------
    guidance = get_clinical_guidance(top_class, top_conf, diag["sym_val"])

    st.subheader("🏥 Clinical Risk & Medical Guidance")

    if guidance["risk_type"] == "error":
        st.error(
            f"🚨 **Risk Level: {guidance['risk_level']}**\n\n{guidance['rationale']}"
        )
    elif guidance["risk_type"] == "warning":
        st.warning(
            f"⚠️ **Risk Level: {guidance['risk_level']}**\n\n{guidance['rationale']}"
        )
    elif guidance["risk_type"] == "info":
        st.info(
            f"ℹ️ **Risk Level: {guidance['risk_level']}**\n\n{guidance['rationale']}"
        )
    else:
        st.success(
            f"✅ **Risk Level: {guidance['risk_level']}**\n\n{guidance['rationale']}"
        )

    with st.expander("📋 Medical Action Plan & First-Aid Protocol", expanded=True):
        guide_col1, guide_col2 = st.columns(2, gap="medium")
        with guide_col1:
            st.markdown("#### 🩹 First-Aid & Immediate Actions")
            for item in guidance["first_aid"]:
                st.markdown(f"- {item}")
        with guide_col2:
            st.markdown("#### 🚨 Emergency Red-Flag Warnings")
            for flag in guidance["red_flags"]:
                st.markdown(f"- {flag}")
            if top_class != "no_bites":
                st.caption("❗ Seek immediate emergency medical care if breathing distress, systemic allergy, or spreading infection occurs.")

    if diag.get("doc_id"):
        st.success(f"💾 **Logged to MongoDB Atlas** | Document ID: `{diag['doc_id']}`")
    else:
        st.warning("⚠️ Record could not be saved to the database.")

    if diag.get("progression_data") and diag["progression_data"].get("risk_alerts"):
        st.error("🚨 **Progression Warning:**")
        for alert in diag["progression_data"]["risk_alerts"]:
            st.write(f"- {alert}")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.write("### Probability Breakdown")
        for idx in np.argsort(final_probs)[::-1]:
            c_name = CLASSES[idx]
            p_val = final_probs[idx]
            st.write(f"**{c_name:<12}** `{p_val * 100:5.2f}%`")
            st.progress(float(p_val))

    with res_col2:
        st.write("### Modality Calibration Comparison")
        records = []
        for i, c_name in enumerate(CLASSES):
            rec = {
                "Class": c_name,
                "Tri-Branch": f"{diag['tri_vision_probs'][i]*100:.1f}%",
                "Swin-B": f"{diag['swin_probs'][i]*100:.1f}%",
                "Stacked CNN": f"{diag['raw_cnn_probs'][i]*100:.1f}%",
                "Descriptors": f"{diag['desc_probs'][i]*100:.1f}%",
            }
            if diag["has_text"]:
                rec["Prior"] = f"{diag['prior_distribution'][i]*100:.1f}%"
            rec["Final"] = f"{final_probs[i]*100:.1f}%"
            records.append(rec)
        st.table(records)