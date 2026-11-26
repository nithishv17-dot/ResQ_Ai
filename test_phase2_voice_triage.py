"""
test_phase2_voice_triage.py
===========================
Comprehensive Test Suite for Phase 2 Feature 4: Voice-Guided 3-Question Triage.

Validates:
    1. Input-Adapter Contract: Output slot strings strictly match Phase 1 options.
    2. Speech-to-Text (STT): CPU int8 faster-whisper transcription.
    3. Text-to-Speech (TTS): Question audio generation & caching.
    4. Circuit Breakers: Graceful fallbacks on missing/corrupt audio or offline states.
    5. Gatekeeper & Fusion Alignment: Verifies that Bayesian prior calculation
       and conflict gatekeeping are unaffected.
"""

import os
import unittest
import numpy as np

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
from app import get_clinical_weights, resolve_multimodal_conflict, CLASSES


class TestVoiceTriageContract(unittest.TestCase):
    """Verifies the Strict Input-Adapter Contract."""

    def test_all_locations_are_valid_phase1_options(self):
        """Ensures every mapped location string is an accepted Phase 1 option."""
        test_inputs = [
            {"q1": "ankle and foot after walking outside"},
            {"q1": "waistband near my belt line and groin"},
            {"q1": "neck, face and arms while sleeping"},
            {"q1": "scalp and behind my ear"},
            {"q1": "on my stomach and chest"},
            {"q1": "unknown mysterious place"},
        ]
        for inp in test_inputs:
            res = map_transcript_to_clinical_slots(inp)
            self.assertIn(
                res["body_location"],
                CLINICAL_LOCATIONS,
                f"Location '{res['body_location']}' not in Phase 1 CLINICAL_LOCATIONS",
            )
            self.assertEqual(res["area"], res["body_location"])

    def test_all_timings_are_valid_phase1_options(self):
        """Ensures every mapped timing string is an accepted Phase 1 option."""
        test_inputs = [
            {"q2": "immediate sharp burning pain minutes ago"},
            {"q2": "woke up with it this morning in bed"},
            {"q2": "noticed it two days later, gradual itching"},
            {"q2": "sometime in the past"},
        ]
        for inp in test_inputs:
            res = map_transcript_to_clinical_slots(inp)
            self.assertIn(
                res["onset_timing"],
                CLINICAL_TIMINGS,
                f"Timing '{res['onset_timing']}' not in Phase 1 CLINICAL_TIMINGS",
            )
            self.assertEqual(res["time"], res["onset_timing"])

    def test_all_symptoms_are_valid_phase1_options(self):
        """Ensures every mapped symptom list contains only accepted Phase 1 options."""
        test_inputs = [
            {"q3": "super itchy with small red bumps and papules"},
            {"q3": "breakfast lunch dinner linear line of bites"},
            {"q3": "expanding red bullseye target ring rash"},
            {"q3": "white blister filled with pus"},
            {"q3": "two distinct fang punctures with necrosis"},
            {"q3": "completely painless, no itching at all"},
            {"q3": "not sure, nothing noticeable"},
        ]
        for inp in test_inputs:
            res = map_transcript_to_clinical_slots(inp)
            self.assertIsInstance(res["hallmark_symptoms"], list)
            for sym in res["hallmark_symptoms"]:
                self.assertIn(
                    sym,
                    CLINICAL_SYMPTOMS,
                    f"Symptom '{sym}' not in Phase 1 CLINICAL_SYMPTOMS",
                )
            self.assertEqual(res["symptoms"], res["hallmark_symptoms"])


class TestSemanticSlotMapper(unittest.TestCase):
    """Verifies natural language keyword and fuzzy slot extraction."""

    def test_prompt_canonical_example(self):
        """
        Tests the exact prompt specification:
        'on my lower ankle after walking in tall grass two hours ago, it's super itchy'
        """
        user_utterance = {
            "full_transcript": "on my lower ankle after walking in tall grass two hours ago, it's super itchy"
        }
        res = map_transcript_to_clinical_slots(user_utterance)

        self.assertEqual(res["body_location"], "Ankles / Feet / Lower Legs")
        self.assertEqual(
            res["onset_timing"], "Delayed reaction over 1–3 days (Gradual itching)"
        )
        self.assertIn(
            "Intense itching with small red papules", res["hallmark_symptoms"]
        )

    def test_separate_slots_extraction(self):
        """Tests individual Q1, Q2, Q3 slot mapping."""
        slots = {
            "q1": "around my waistband and groin area",
            "q2": "immediate burning sensation within minutes",
            "q3": "white fluid-filled blister with a pustule",
        }
        res = map_transcript_to_clinical_slots(slots)

        self.assertEqual(res["body_location"], "Waistband / Groin / Skin Folds")
        self.assertEqual(
            res["onset_timing"],
            "Immediate / within minutes (Instant burning or sharp pain)",
        )
        self.assertEqual(
            res["hallmark_symptoms"], ["Pustule or white fluid-filled blister"]
        )

    def test_negation_handling(self):
        """Verifies that 'no itching' does not trigger 'Intense itching'."""
        slots = {
            "q1": "scalp and behind ear",
            "q2": "woke up this morning",
            "q3": "it is not itchy, completely painless bite",
        }
        res = map_transcript_to_clinical_slots(slots)

        self.assertIn("No itching / completely painless", res["hallmark_symptoms"])
        self.assertNotIn(
            "Intense itching with small red papules", res["hallmark_symptoms"]
        )

    def test_fallback_defaults_on_unrecognized_input(self):
        """Verifies that empty or unrecognizable text cleanly defaults."""
        slots = {"q1": "asdfghjkl", "q2": "xyz123", "q3": "blah blah"}
        res = map_transcript_to_clinical_slots(slots)

        self.assertEqual(res["body_location"], "Select / Not specified")
        self.assertEqual(res["onset_timing"], "Select / Not specified")
        self.assertEqual(res["hallmark_symptoms"], ["None / Unsure"])


class TestTextToSpeech(unittest.TestCase):
    """Verifies TTS synthesis and audio prompt generation."""

    def test_generate_all_triage_questions(self):
        """Verifies that audio is generated or loaded from cache for all 3 questions."""
        for q_key, q_text in TRIAGE_QUESTIONS.items():
            audio_bytes = generate_question_audio(q_text)
            self.assertIsNotNone(
                audio_bytes, f"Audio bytes should not be None for {q_key}"
            )
            self.assertGreater(
                len(audio_bytes), 1000, f"Audio bytes too short for {q_key}"
            )

    def test_tts_circuit_breaker_on_empty(self):
        """Verifies that empty string returns None without raising an exception."""
        self.assertIsNone(generate_question_audio(""))
        self.assertIsNone(generate_question_audio("   "))


class TestSpeechToText(unittest.TestCase):
    """Verifies STT transcription using faster-whisper CPU int8."""

    def test_transcribe_patient_audio_with_sample(self):
        """Verifies transcription of known audio file (q1.mp3)."""
        cache_path = os.path.join(
            os.path.dirname(__file__), "voice_cache", "q1.mp3"
        )
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                sample_bytes = f.read()
            text = transcribe_patient_audio(sample_bytes)
            self.assertIsInstance(text, str)
            self.assertIn("body", text)
            self.assertIn("bite", text)

    def test_stt_circuit_breaker_on_corrupt_or_empty(self):
        """Verifies that corrupt or empty audio returns empty string without crashing."""
        self.assertEqual(transcribe_patient_audio(None), "")
        self.assertEqual(transcribe_patient_audio(b""), "")
        self.assertEqual(transcribe_patient_audio(b"corrupted raw random bytes 12345"), "")


class TestBayesianFusionNonInterference(unittest.TestCase):
    """Verifies zero core coupling and proper Bayesian integration with app.py."""

    def test_mapped_slots_integrate_with_clinical_weights(self):
        """Verifies that get_clinical_weights properly processes mapped slots."""
        user_utterance = {
            "full_transcript": "on my lower ankle after walking in tall grass two hours ago, it's super itchy"
        }
        res = map_transcript_to_clinical_slots(user_utterance)

        has_text, prior_vec = get_clinical_weights(
            location=res["body_location"],
            timing=res["onset_timing"],
            symptoms=res["hallmark_symptoms"],
        )

        self.assertTrue(has_text)
        self.assertEqual(len(prior_vec), len(CLASSES))
        self.assertAlmostEqual(float(np.sum(prior_vec)), 1.0, places=5)

        # Fleas, chiggers, ants should receive elevated prior on lower leg
        flea_idx = CLASSES.index("fleas")
        self.assertGreater(prior_vec[flea_idx], 1.0 / len(CLASSES))

    def test_conflict_gatekeeper_threshold_intact(self):
        """
        Verifies that resolve_multimodal_conflict suppresses priors if
        cosine similarity < 0.35 threshold.
        """
        # Contrived scenario: strong vision for spiders, but text prior for ticks
        cnn_probs = np.zeros(len(CLASSES), dtype=np.float32)
        cnn_probs[CLASSES.index("spiders")] = 0.90
        cnn_probs[CLASSES.index("mosquitos")] = 0.10

        prior_probs = np.zeros(len(CLASSES), dtype=np.float32)
        prior_probs[CLASSES.index("ticks")] = 0.90
        prior_probs[CLASSES.index("bed_bugs")] = 0.10

        is_conflict, final_probs, msg = resolve_multimodal_conflict(
            cnn_probs, prior_probs, confidence_threshold=0.50, similarity_threshold=0.35
        )

        self.assertTrue(is_conflict)
        self.assertIn("Conflict Detected", msg)
        # Should revert cleanly to cnn_probs
        np.testing.assert_array_almost_equal(final_probs, cnn_probs)


    def test_handy_dictation_keys(self):
        """Verifies mapping when text is injected via Handy into quick_dictation or unified."""
        test_inputs = [
            {"unified": "on my lower ankle after walking in tall grass two hours ago, it's super itchy"},
            {"quick_dictation": "waistband near my belt line, immediate burning pain, white pus blister"},
        ]
        res1 = map_transcript_to_clinical_slots(test_inputs[0])
        self.assertEqual(res1["body_location"], "Ankles / Feet / Lower Legs")
        self.assertEqual(res1["onset_timing"], "Delayed reaction over 1–3 days (Gradual itching)")
        self.assertIn("Intense itching with small red papules", res1["hallmark_symptoms"])

        res2 = map_transcript_to_clinical_slots(test_inputs[1])
        self.assertEqual(res2["body_location"], "Waistband / Groin / Skin Folds")
        self.assertEqual(res2["onset_timing"], "Immediate / within minutes (Instant burning or sharp pain)")
        self.assertIn("Pustule or white fluid-filled blister", res2["hallmark_symptoms"])


if __name__ == "__main__":
    unittest.main()
