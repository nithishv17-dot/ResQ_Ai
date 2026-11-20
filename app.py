import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from db_manager import BiteDBManager
from download_model import ensure_model

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
MODEL_PATH = os.getenv("MODEL_PATH", "models/bug_bite_8class_stacked_model.keras")

# ---------------------------------------------------------
# Resource Initialization
# ---------------------------------------------------------
@st.cache_resource
def load_classifier():
    ensure_model(MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at path: `{MODEL_PATH}`")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_db():
    return BiteDBManager()

model = load_classifier()
db = load_db()

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

    if db.is_connected:
        st.success("🟢 MongoDB Atlas Connected")
    else:
        st.error("🔴 MongoDB Disconnected")

with st.form("diagnosis_form", clear_on_submit=False):
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

    with col_text:
        st.subheader("2. Clinical Prior Context")
        loc_val = st.selectbox(
            "Anatomical Location:",
            [
                "Select / Not specified",
                "Ankles / Feet / Lower Legs",
                "Waistband / Groin / Skin Folds",
                "Neck / Face / Arms (Exposed while sleeping)",
                "Scalp / Behind Ears / Armpits",
                "General / Other Area",
            ],
        )

        time_val = st.selectbox(
            "Onset Timing:",
            [
                "Select / Not specified",
                "Immediate / within minutes (Instant burning or sharp pain)",
                "Woke up with bites / Noticed in the morning",
                "Delayed reaction over 1–3 days (Gradual itching)",
            ],
        )

        sym_val = st.multiselect(
            "Observed Clinical Signs:",
            [
                "None / Unsure",
                "Linear track / 'Breakfast-Lunch-Dinner' clustered line",
                "Intense itching with small red papules",
                "Expanding bullseye / target ring rash",
                "Pustule or white fluid-filled blister",
                "Two distinct fang marks / Central necrosis",
                "No itching / completely painless",
            ],
        )

    submit_button = st.form_submit_button("🚀 Run Diagnosis", use_container_width=True)

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
            input_tensor = preprocess_image(image_obj)
            raw_cnn_probs = model.predict(input_tensor, verbose=0)[0]

            has_text, prior_distribution = get_clinical_weights(loc_val, time_val, sym_val)

            if has_text:
                is_conflict, final_probs, status_msg = resolve_multimodal_conflict(
                    raw_cnn_probs, prior_distribution
                )
                if is_conflict:
                    st.warning(f"⚠️ {status_msg}")
                else:
                    st.info(f"✅ {status_msg}")
            else:
                final_probs = raw_cnn_probs
                st.caption("Executed in **Pure Vision Mode** (no clinical metadata provided).")

            top_idx = int(np.argmax(final_probs))
            top_class = CLASSES[top_idx]
            top_conf = float(final_probs[top_idx] * 100)

            prob_dict = {CLASSES[i]: float(final_probs[i]) for i in range(len(CLASSES))}

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

        st.markdown("---")
        st.subheader("📊 Diagnostic Verdict")
        st.success(
            f"**Predicted Class:** `{top_class.upper()}` | **Confidence:** {top_conf:.2f}% | **Source:** {input_source_label}"
        )

        # -----------------------------------------------------
        # Clinical Risk Assessment & Medical Action Plan
        # -----------------------------------------------------
        guidance = get_clinical_guidance(top_class, top_conf, sym_val)

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

        if doc_id:
            st.success(f"💾 **Logged to MongoDB Atlas** | Document ID: `{doc_id}`")
        else:
            st.warning("⚠️ Record could not be saved to the database.")

        if progression_data and progression_data.get("risk_alerts"):
            st.error("🚨 **Progression Warning:**")
            for alert in progression_data["risk_alerts"]:
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
            if has_text:
                st.write("### Modality Calibration Comparison")
                records = []
                for i, c_name in enumerate(CLASSES):
                    records.append({
                        "Class": c_name,
                        "Vision CNN": f"{raw_cnn_probs[i]*100:.1f}%",
                        "Clinical Prior": f"{prior_distribution[i]*100:.1f}%",
                        "Final Calibrated": f"{final_probs[i]*100:.1f}%",
                    })
                st.table(records)