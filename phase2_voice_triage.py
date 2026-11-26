"""
phase2_voice_triage.py
======================
Phase 2 Feature 4: Voice-Guided 3-Question Triage for ResQ_Ai.

Dual Voice Architecture:
    1. Primary Workflow: Support for Handy (open-source, offline OS-level Whisper/
       Parakeet desktop dictation app). Users press Handy's global hotkey to dictate
       answers directly into focused input fields with ZERO Python audio streaming
       or RAM overhead.
    2. Fallback Workflow: Optional in-browser microphone capture (st.audio_input +
       faster-whisper CPU int8) safely isolated inside circuit-breaker try/except
       blocks for evaluators without Handy installed.

Architectural Guardrails:
    - Strict Input-Adapter Contract: Output slot strings strictly match canonical
      Phase 1 options (CLINICAL_LOCATIONS, CLINICAL_TIMINGS, CLINICAL_SYMPTOMS).
    - Zero Core Coupling: Never modifies vision models, Bayesian fusion formulas,
      or the Cosine Similarity Conflict Gatekeeper (< 0.35 threshold).
    - Zero Mandatory Audio Dependencies: Gracefully degrades to keyboard/text entry
      if audio libraries or hardware are absent.
"""

import io
import os
import re
import tempfile
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any

# Configure module logger
logger = logging.getLogger("phase2_voice_triage")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Canonical Clinical Constants (Exact Phase 1 Option Strings)
# ---------------------------------------------------------------------------

CLINICAL_LOCATIONS: List[str] = [
    "Select / Not specified",
    "Ankles / Feet / Lower Legs",
    "Waistband / Groin / Skin Folds",
    "Neck / Face / Arms (Exposed while sleeping)",
    "Scalp / Behind Ears / Armpits",
    "General / Other Area",
]

CLINICAL_TIMINGS: List[str] = [
    "Select / Not specified",
    "Immediate / within minutes (Instant burning or sharp pain)",
    "Woke up with bites / Noticed in the morning",
    "Delayed reaction over 1–3 days (Gradual itching)",
]

CLINICAL_SYMPTOMS: List[str] = [
    "None / Unsure",
    "Linear track / 'Breakfast-Lunch-Dinner' clustered line",
    "Intense itching with small red papules",
    "Expanding bullseye / target ring rash",
    "Pustule or white fluid-filled blister",
    "Two distinct fang marks / Central necrosis",
    "No itching / completely painless",
]

TRIAGE_QUESTIONS: Dict[str, str] = {
    "q1_location": "Where on your body is the bite or rash located?",
    "q2_timing": "When did you first notice the bite or symptoms?",
    "q3_symptoms": (
        "What specific symptoms are you feeling (itching, burning, swelling, "
        "ring-shaped rash)?"
    ),
}

# Local disk cache directory for offline resilience
CACHE_DIR = os.path.join(os.path.dirname(__file__), "voice_cache")

# In-memory TTS byte cache
_AUDIO_PROMPT_CACHE: Dict[str, bytes] = {}

# Lazy singleton faster-whisper model
_WHISPER_MODEL = None


# ---------------------------------------------------------------------------
# Optional Audio Libraries Loader (Zero Mandatory Dependency Guarantee)
# ---------------------------------------------------------------------------

def is_whisper_available() -> bool:
    """Checks whether faster-whisper is installed and importable."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def is_tts_available() -> bool:
    """Checks whether gTTS is installed and importable."""
    try:
        import gtts  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 1. Text-to-Speech (TTS): generate_question_audio
# ---------------------------------------------------------------------------

def generate_question_audio(question_text: str, lang: str = "en") -> Optional[bytes]:
    """
    Synthesizes clear spoken audio prompts for clinical triage questions.
    Checks disk cache, in-memory cache, and falls back to dynamic gTTS.
    Wrapped in circuit-breaker try/except blocks to guarantee zero-crash execution.
    """
    if not question_text or not question_text.strip():
        return None

    norm_text = question_text.strip()

    # 1. In-memory cache
    if norm_text in _AUDIO_PROMPT_CACHE:
        return _AUDIO_PROMPT_CACHE[norm_text]

    # 2. Local disk cache (voice_cache/)
    cache_file_map = {
        TRIAGE_QUESTIONS["q1_location"]: "q1.mp3",
        TRIAGE_QUESTIONS["q2_timing"]: "q2.mp3",
        TRIAGE_QUESTIONS["q3_symptoms"]: "q3.mp3",
    }
    mapped_filename = cache_file_map.get(norm_text)
    if mapped_filename:
        disk_path = os.path.join(CACHE_DIR, mapped_filename)
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "rb") as f:
                    audio_bytes = f.read()
                _AUDIO_PROMPT_CACHE[norm_text] = audio_bytes
                return audio_bytes
            except Exception as e:
                logger.warning(f"Error reading disk cache {disk_path}: {e}")

    # 3. Dynamic synthesis via gTTS (optional fallback)
    try:
        from gtts import gTTS

        buffer = io.BytesIO()
        tts = gTTS(text=norm_text, lang=lang, tld="com", slow=False)
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.getvalue()
        _AUDIO_PROMPT_CACHE[norm_text] = audio_bytes
        return audio_bytes
    except Exception as e:
        logger.warning(f"TTS synthesis circuit breaker active for '{norm_text[:30]}...': {e}")
        return None


# ---------------------------------------------------------------------------
# 2. Speech-to-Text (STT): transcribe_patient_audio
# ---------------------------------------------------------------------------

def _get_whisper_model(model_size: str = "tiny.en"):
    """Initializes or returns cached faster-whisper WhisperModel singleton."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster-whisper model '{model_size}' on CPU int8...")
            _WHISPER_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as e:
            logger.error(f"Failed to initialize faster-whisper: {e}")
            raise e
    return _WHISPER_MODEL


def transcribe_patient_audio(
    audio_bytes: Optional[bytes],
    model_size: str = "tiny.en",
) -> str:
    """
    Transcribes patient vocal recordings into clean lowercase text using faster-whisper.
    Employs CPU int8 execution for lightweight low-RAM footprint.

    Circuit Breaker:
        Gracefully handles corrupted audio, unsupported encodings, empty buffers,
        or missing libraries without throwing unhandled exceptions.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return ""

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        model = _get_whisper_model(model_size=model_size)
        segments, info = model.transcribe(
            temp_path,
            beam_size=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
        )

        segment_texts = [seg.text for seg in segments]
        raw_text = " ".join(segment_texts).strip()
        clean_text = re.sub(r"\s+", " ", raw_text).lower().strip()
        return clean_text

    except Exception as e:
        logger.warning(f"Whisper transcription circuit breaker triggered: {e}")
        return ""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 3. Domain Lexicon & Semantic Parsing
# ---------------------------------------------------------------------------

LOCATION_LEXICON = {
    "Ankles / Feet / Lower Legs": [
        "ankle", "ankles", "foot", "feet", "leg", "legs", "lower leg", "lower legs",
        "calf", "calves", "shin", "shins", "toe", "toes", "heel", "heels",
        "sock", "socks", "sock line", "sockline", "grass", "tall grass", "lawn",
    ],
    "Waistband / Groin / Skin Folds": [
        "waist", "waistband", "groin", "crotch", "hip", "hips", "skin fold", "skinfolds",
        "belt", "beltline", "belt line", "underwear", "underwear line",
        "thigh", "thighs", "inner thigh", "butt", "buttock", "buttocks", "glute",
        "crease", "genital", "pubic",
    ],
    "Neck / Face / Arms (Exposed while sleeping)": [
        "neck", "face", "arm", "arms", "hand", "hands", "wrist", "wrists",
        "shoulder", "shoulders", "cheek", "cheeks", "forehead", "chin",
        "forearm", "forearms", "elbow", "elbows", "finger", "fingers",
        "sleeping", "in bed", "pillow", "mattress",
    ],
    "Scalp / Behind Ears / Armpits": [
        "scalp", "head", "hair", "hairline", "ear", "ears", "behind ear",
        "behind ears", "behind the ear", "armpit", "armpits", "underarm",
        "underarms", "axilla",
    ],
    "General / Other Area": [
        "back", "upper back", "lower back", "chest", "stomach", "belly",
        "abdomen", "torso", "trunk", "body", "all over", "multiple areas",
        "everywhere", "general",
    ],
}

TIMING_LEXICON = {
    "Immediate / within minutes (Instant burning or sharp pain)": [
        "immediate", "immediately", "instant", "instantly", "minute", "minutes",
        "sharp pain", "burning", "sting", "stung", "sharp", "just now", "sudden",
        "suddenly", "right away", "right now", "few seconds", "seconds ago",
        "at once", "felt it happen",
    ],
    "Woke up with bites / Noticed in the morning": [
        "woke up", "wake up", "woken up", "morning", "in the morning", "this morning",
        "in bed", "slept", "sleeping", "night", "overnight", "waking up",
        "got out of bed", "over night",
    ],
    "Delayed reaction over 1–3 days (Gradual itching)": [
        "delayed", "day", "days", "yesterday", "hours ago", "hour ago", "two hours ago",
        "few hours ago", "couple hours", "gradual", "gradually", "few days",
        "later", "couple days", "past few days", "1 day", "2 days", "3 days",
        "one day", "two days", "three days", "hike yesterday", "after hiking",
        "tall grass two hours ago", "a while ago", "started slowly",
    ],
}

SYMPTOM_LEXICON = {
    "Linear track / 'Breakfast-Lunch-Dinner' clustered line": [
        "linear", "line", "straight line", "track", "row", "in a row", "cluster",
        "clustered", "breakfast lunch dinner", "three bites", "series of bites",
        "zigzag", "grouped together", "line of bites",
    ],
    "Intense itching with small red papules": [
        "itch", "itchy", "itching", "super itchy", "intense itching", "very itchy",
        "papule", "papules", "red bump", "red bumps", "small bump", "small bumps",
        "bump", "bumps", "scratch", "scratching", "pruritic", "itchiness",
        "hives", "welts",
    ],
    "Expanding bullseye / target ring rash": [
        "bullseye", "bull's eye", "bulls eye", "bull eye", "target", "ring",
        "rings", "annular", "expanding", "circle", "circular", "red ring",
        "target rash", "expanding rash", "concentric", "spreading ring",
    ],
    "Pustule or white fluid-filled blister": [
        "pustule", "pustules", "pus", "blister", "blisters", "fluid", "fluid filled",
        "fluid-filled", "white spot", "white spots", "white head", "whitehead",
        "vesicle", "vesicles", "pimple", "pimple-like",
    ],
    "Two distinct fang marks / Central necrosis": [
        "fang", "fangs", "fang mark", "fang marks", "two marks", "two dots",
        "two punctures", "two holes", "puncture", "punctures", "black center",
        "dark center", "necrosis", "necrotic", "dying tissue", "tissue death",
        "purple center", "ulcer",
    ],
    "No itching / completely painless": [
        "no itch", "no itching", "not itchy", "painless", "no pain", "doesn't hurt",
        "does not hurt", "didn't hurt", "did not hurt", "completely painless",
        "painless bite", "unnoticed", "felt nothing", "no sensation",
    ],
}


def _calculate_lexicon_score(text: str, keywords: List[str]) -> Tuple[int, float]:
    """Computes exact keyword occurrence count and best fuzzy token similarity."""
    text_lower = f" {text.lower()} "
    exact_hits = 0
    best_fuzzy = 0.0

    for kw in keywords:
        kw_pattern = rf"\b{re.escape(kw)}\b"
        matches = re.findall(kw_pattern, text_lower)
        if matches:
            exact_hits += len(matches) * (2 if " " in kw else 1)
        else:
            if kw in text_lower:
                exact_hits += 1

    words = [w for w in re.split(r"\W+", text_lower) if len(w) > 3]
    for kw in keywords:
        kw_words = [w for w in re.split(r"\W+", kw) if len(w) > 3]
        for w in words:
            for kw_w in kw_words:
                ratio = SequenceMatcher(None, w, kw_w).ratio()
                if ratio > best_fuzzy:
                    best_fuzzy = ratio

    return exact_hits, best_fuzzy


def _map_location_slot(corpus: str) -> str:
    """Maps free-form text to one of CLINICAL_LOCATIONS."""
    if not corpus.strip():
        return "Select / Not specified"

    best_category = "Select / Not specified"
    highest_score = 0.0

    for category, keywords in LOCATION_LEXICON.items():
        hits, fuzzy = _calculate_lexicon_score(corpus, keywords)
        score = hits * 2.0 + (fuzzy if fuzzy >= 0.80 else 0.0)
        if score > highest_score and score >= 1.0:
            highest_score = score
            best_category = category

    return best_category


def _map_timing_slot(corpus: str) -> str:
    """Maps free-form text to one of CLINICAL_TIMINGS."""
    if not corpus.strip():
        return "Select / Not specified"

    best_category = "Select / Not specified"
    highest_score = 0.0

    for category, keywords in TIMING_LEXICON.items():
        hits, fuzzy = _calculate_lexicon_score(corpus, keywords)
        score = hits * 2.0 + (fuzzy if fuzzy >= 0.80 else 0.0)
        if score > highest_score and score >= 1.0:
            highest_score = score
            best_category = category

    return best_category


def _map_symptoms_slot(corpus: str) -> List[str]:
    """
    Maps free-form text to a list of observed symptoms from CLINICAL_SYMPTOMS.
    Handles negation (e.g., 'not itchy') to avoid false positives on 'itchy'.
    """
    if not corpus.strip():
        return ["None / Unsure"]

    corpus_lower = corpus.lower()

    none_patterns = [
        "none", "unsure", "not sure", "don't know", "do not know",
        "can't tell", "cannot tell", "no idea", "nothing",
    ]
    for p in none_patterns:
        if re.search(rf"\b{re.escape(p)}\b", corpus_lower) and len(corpus_lower.split()) <= 4:
            return ["None / Unsure"]

    matched_symptoms: List[str] = []

    painless_hits, _ = _calculate_lexicon_score(
        corpus_lower, SYMPTOM_LEXICON["No itching / completely painless"]
    )
    is_painless = painless_hits > 0

    if is_painless:
        matched_symptoms.append("No itching / completely painless")

    for category, keywords in SYMPTOM_LEXICON.items():
        if category == "No itching / completely painless":
            continue

        if category == "Intense itching with small red papules" and is_painless:
            continue

        hits, fuzzy = _calculate_lexicon_score(corpus_lower, keywords)
        if hits > 0 or fuzzy >= 0.85:
            matched_symptoms.append(category)

    if not matched_symptoms:
        return ["None / Unsure"]

    return matched_symptoms


# ---------------------------------------------------------------------------
# 4. Semantic Slot Mapper: map_transcript_to_clinical_slots
# ---------------------------------------------------------------------------

def map_transcript_to_clinical_slots(
    raw_answers_dict: Dict[str, Any],
    valid_options_dict: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Parses free-form spoken transcripts from Handy (or browser mic) into exact
    categorical dropdown strings expected by the Phase 1 Bayesian engine.

    Strict Contract:
        Outputs ONLY valid options from CLINICAL_LOCATIONS, CLINICAL_TIMINGS,
        and CLINICAL_SYMPTOMS. Defaults to unmapped choices if unrecognized.

    Args:
        raw_answers_dict: Dictionary containing raw text responses.
        valid_options_dict: Optional override choices for validation.

    Returns:
        Standardized dictionary:
        {
            "body_location": str,
            "onset_timing": str,
            "hallmark_symptoms": List[str],
            "area": str,
            "time": str,
            "symptoms": List[str],
            "raw_answers": dict,
        }
    """
    valid_locs = (
        valid_options_dict.get("body_location", CLINICAL_LOCATIONS)
        if valid_options_dict
        else CLINICAL_LOCATIONS
    )
    valid_times = (
        valid_options_dict.get("onset_timing", CLINICAL_TIMINGS)
        if valid_options_dict
        else CLINICAL_TIMINGS
    )
    valid_syms = (
        valid_options_dict.get("hallmark_symptoms", CLINICAL_SYMPTOMS)
        if valid_options_dict
        else CLINICAL_SYMPTOMS
    )

    if not raw_answers_dict or not isinstance(raw_answers_dict, dict):
        return {
            "body_location": "Select / Not specified",
            "onset_timing": "Select / Not specified",
            "hallmark_symptoms": ["None / Unsure"],
            "area": "Select / Not specified",
            "time": "Select / Not specified",
            "symptoms": ["None / Unsure"],
            "raw_answers": {},
        }

    # Extract slot-specific text or full corpus
    text_loc = str(
        raw_answers_dict.get("body_location")
        or raw_answers_dict.get("q1_location")
        or raw_answers_dict.get("q1")
        or raw_answers_dict.get("area")
        or raw_answers_dict.get("location")
        or ""
    ).strip()

    text_time = str(
        raw_answers_dict.get("onset_timing")
        or raw_answers_dict.get("q2_timing")
        or raw_answers_dict.get("q2")
        or raw_answers_dict.get("time")
        or raw_answers_dict.get("timing")
        or ""
    ).strip()

    text_sym = str(
        raw_answers_dict.get("hallmark_symptoms")
        or raw_answers_dict.get("q3_symptoms")
        or raw_answers_dict.get("q3")
        or raw_answers_dict.get("symptoms")
        or ""
    ).strip()

    full_corpus = " ".join([text_loc, text_time, text_sym]).strip()
    for alt_key in ["full_transcript", "quick_dictation", "unified", "single"]:
        if not full_corpus and alt_key in raw_answers_dict:
            full_corpus = str(raw_answers_dict[alt_key]).strip()

    # 1. Map Location
    mapped_loc = _map_location_slot(text_loc)
    if mapped_loc == "Select / Not specified" and full_corpus != text_loc:
        mapped_loc = _map_location_slot(full_corpus)
    if mapped_loc not in valid_locs:
        mapped_loc = "Select / Not specified"

    # 2. Map Timing
    mapped_time = _map_timing_slot(text_time)
    if mapped_time == "Select / Not specified" and full_corpus != text_time:
        mapped_time = _map_timing_slot(full_corpus)
    if mapped_time not in valid_times:
        mapped_time = "Select / Not specified"

    # 3. Map Symptoms
    mapped_sym = _map_symptoms_slot(text_sym)
    if mapped_sym == ["None / Unsure"] and full_corpus != text_sym:
        mapped_sym = _map_symptoms_slot(full_corpus)
    mapped_sym = [s for s in mapped_sym if s in valid_syms]
    if not mapped_sym:
        mapped_sym = ["None / Unsure"]

    return {
        "body_location": mapped_loc,
        "onset_timing": mapped_time,
        "hallmark_symptoms": mapped_sym,
        "area": mapped_loc,
        "time": mapped_time,
        "symptoms": mapped_sym,
        "raw_answers": raw_answers_dict,
    }


# ---------------------------------------------------------------------------
# 5. Guided Dictation UI: render_handy_voice_triage_ui
# ---------------------------------------------------------------------------

def render_handy_voice_triage_ui(
    valid_options_dict: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Renders the Hands-Free Voice Triage UI component inside Streamlit.

    Dual Architecture:
        1. Primary: Handy OS-Level Speech Dictation. Displays focused text inputs
           where the user triggers their Handy desktop hotkey to dictate speech directly
           with ZERO extra Python RAM or audio latency.
        2. Fallback: Browser Mic Fallback using st.audio_input + faster-whisper,
           isolated safely inside a circuit-breaker try/except block.

    Returns:
        Dictionary containing mapped slots and activation state.
    """
    import hashlib
    import streamlit as st

    # Initialize state for Handy text inputs
    if "handy_q1_text" not in st.session_state:
        st.session_state["handy_q1_text"] = ""
    if "handy_q2_text" not in st.session_state:
        st.session_state["handy_q2_text"] = ""
    if "handy_q3_text" not in st.session_state:
        st.session_state["handy_q3_text"] = ""
    if "handy_unified_text" not in st.session_state:
        st.session_state["handy_unified_text"] = ""
    if "browser_mic_hashes" not in st.session_state:
        st.session_state["browser_mic_hashes"] = {}

    with st.expander("🎙️ Hands-Free Voice Triage (Handy AI)", expanded=False):
        st.markdown(
            "💡 **Handy Dictation Mode (Recommended):** Click into any field below, press your "
            "**Handy global hotkey** (e.g. `Ctrl+Space`), speak naturally, and release. "
            "Speech will inject directly with zero lag."
        )

        tab_questions, tab_quick, tab_fallback = st.tabs([
            "📋 3-Question Dictation",
            "⚡ Quick Single Dictation",
            "🎙️ Browser Mic Fallback (No Handy)",
        ])

        voice_changed = False

        # --- Tab 1: 3-Question Structured Dictation ---
        with tab_questions:
            q1_in = st.text_input(
                "📍 Q1 (Location): " + TRIAGE_QUESTIONS["q1_location"],
                value=st.session_state["handy_q1_text"],
                placeholder="Dictate with Handy (e.g., lower ankle after walking in tall grass)",
                key="input_handy_q1",
            )
            if q1_in != st.session_state["handy_q1_text"]:
                st.session_state["handy_q1_text"] = q1_in
                voice_changed = True

            q2_in = st.text_input(
                "⏱️ Q2 (Timing): " + TRIAGE_QUESTIONS["q2_timing"],
                value=st.session_state["handy_q2_text"],
                placeholder="Dictate with Handy (e.g., about two hours ago / woke up this morning)",
                key="input_handy_q2",
            )
            if q2_in != st.session_state["handy_q2_text"]:
                st.session_state["handy_q2_text"] = q2_in
                voice_changed = True

            q3_in = st.text_input(
                "🩺 Q3 (Symptoms): " + TRIAGE_QUESTIONS["q3_symptoms"],
                value=st.session_state["handy_q3_text"],
                placeholder="Dictate with Handy (e.g., super itchy with small red bumps)",
                key="input_handy_q3",
            )
            if q3_in != st.session_state["handy_q3_text"]:
                st.session_state["handy_q3_text"] = q3_in
                voice_changed = True

        # --- Tab 2: Unified Quick Single Dictation ---
        with tab_quick:
            st.caption("Dictate your entire clinical context in one complete sentence:")
            unified_in = st.text_area(
                "Unified Patient Utterance:",
                value=st.session_state["handy_unified_text"],
                placeholder=(
                    "e.g., 'On my lower ankle after walking in tall grass two hours ago, "
                    "it is super itchy with red bumps'"
                ),
                key="input_handy_unified",
                height=85,
            )
            if unified_in != st.session_state["handy_unified_text"]:
                st.session_state["handy_unified_text"] = unified_in
                voice_changed = True

        # --- Tab 3: In-Browser Microphone Fallback (Circuit Breaker) ---
        with tab_fallback:
            st.caption(
                "No Handy desktop app installed? Record directly through your browser microphone "
                "using in-memory Whisper transcription:"
            )

            fb_col1, fb_col2 = st.columns([1, 1], gap="small")
            with fb_col1:
                # Optional TTS question prompt audio
                try:
                    q1_audio = generate_question_audio(TRIAGE_QUESTIONS["q1_location"])
                    if q1_audio:
                        st.audio(q1_audio, format="audio/mp3")
                except Exception as e:
                    logger.warning(f"Fallback audio prompt circuit breaker: {e}")

                fallback_mic = st.audio_input("Browser Mic Capture:", key="fallback_browser_mic")
                if fallback_mic is not None:
                    fb_bytes = fallback_mic.getvalue()
                    fb_hash = hashlib.sha256(fb_bytes).hexdigest()
                    if st.session_state["browser_mic_hashes"].get("mic") != fb_hash:
                        with st.spinner("Transcribing browser audio (faster-whisper CPU int8)..."):
                            fb_text = transcribe_patient_audio(fb_bytes)
                            st.session_state["browser_mic_hashes"]["mic"] = fb_hash
                            if fb_text:
                                st.session_state["handy_unified_text"] = fb_text
                                voice_changed = True

            with fb_col2:
                if st.session_state["handy_unified_text"]:
                    st.info(f"🗣️ **Captured Transcript:**\n*\"{st.session_state['handy_unified_text']}\"*")
                else:
                    st.caption("Press record to speak if you do not have Handy installed.")

        # Aggregate inputs
        raw_dict = {
            "q1": st.session_state["handy_q1_text"],
            "q2": st.session_state["handy_q2_text"],
            "q3": st.session_state["handy_q3_text"],
            "unified": st.session_state["handy_unified_text"],
        }

        has_any_input = any(v.strip() for v in raw_dict.values())
        mapped = map_transcript_to_clinical_slots(raw_dict, valid_options_dict)

        if has_any_input:
            st.markdown("---")
            st.markdown("##### 🎯 Semantic Slot Mapping Summary (Handy AI)")
            sym_str = (
                ", ".join(mapped["hallmark_symptoms"])
                if mapped["hallmark_symptoms"]
                else "None / Unsure"
            )
            st.success(
                f"📍 **Location:** `{mapped['body_location']}`\n\n"
                f"⏱️ **Timing:** `{mapped['onset_timing']}`\n\n"
                f"🩺 **Symptoms:** `{sym_str}`"
            )

            if st.button("🔄 Reset Handy Dictation", use_container_width=True):
                st.session_state["handy_q1_text"] = ""
                st.session_state["handy_q2_text"] = ""
                st.session_state["handy_q3_text"] = ""
                st.session_state["handy_unified_text"] = ""
                st.session_state["browser_mic_hashes"] = {}
                st.session_state["sel_location_key"] = "Select / Not specified"
                st.session_state["sel_timing_key"] = "Select / Not specified"
                st.session_state["sel_symptoms_key"] = []
                st.rerun()

        mapped["is_active"] = has_any_input
        mapped["has_changed"] = voice_changed
        return mapped


# ---------------------------------------------------------------------------
# Self-Test Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Testing phase2_voice_triage.py with Handy Dictation Support")
    print("=" * 65)

    test_cases = [
        {
            "desc": "Handy Prompt Specification Example",
            "input": {
                "unified": "on my lower ankle after walking in tall grass two hours ago, it's super itchy"
            },
            "expected_loc": "Ankles / Feet / Lower Legs",
            "expected_time": "Delayed reaction over 1–3 days (Gradual itching)",
            "expected_sym": ["Intense itching with small red papules"],
        },
        {
            "desc": "Handy 3-Question Structured Example",
            "input": {
                "q1": "waistband near my belt line",
                "q2": "immediate burning sensation",
                "q3": "white blister filled with pus",
            },
            "expected_loc": "Waistband / Groin / Skin Folds",
            "expected_time": "Immediate / within minutes (Instant burning or sharp pain)",
            "expected_sym": ["Pustule or white fluid-filled blister"],
        },
        {
            "desc": "Painless Tick Bullseye Example",
            "input": {
                "unified": "target bullseye ring rash behind ear, completely painless no itching"
            },
            "expected_loc": "Scalp / Behind Ears / Armpits",
            "expected_time": "Select / Not specified",
            "expected_sym": [
                "No itching / completely painless",
                "Expanding bullseye / target ring rash",
            ],
        },
    ]

    all_passed = True
    for i, tc in enumerate(test_cases, 1):
        res = map_transcript_to_clinical_slots(tc["input"])
        loc_ok = res["body_location"] == tc["expected_loc"]
        time_ok = res["onset_timing"] == tc["expected_time"]
        sym_ok = set(res["hallmark_symptoms"]) == set(tc["expected_sym"])
        passed = loc_ok and time_ok and sym_ok
        if not passed:
            all_passed = False
        print(f"\nTest {i} [{tc['desc']}]: {'PASS' if passed else 'FAIL'}")
        print(f"  Location: '{res['body_location']}' [OK={loc_ok}]")
        print(f"  Timing:   '{res['onset_timing']}' [OK={time_ok}]")
        print(f"  Symptoms: {res['hallmark_symptoms']} [OK={sym_ok}]")

    print("\n" + "=" * 65)
    print(f"Self-Test Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 65)
