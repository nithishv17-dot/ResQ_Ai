"""
test_image_questions.py
=======================
Comprehensive Unit and Integration Test Suite for Phase 2 Feature 5:
Image-Based Dynamic Questioning (Visual-Triage Active Clarification).

Validates:
    1. High-confidence bypass when predictions exceed confidence thresholds.
    2. Active clarification trigger on narrow margins or high entropy.
    3. Mutual Information Gain (IG) calculation and visual biomarker boosting.
    4. Exact Bayesian posterior updates for 'Yes', 'No', and 'Not sure' answers.
    5. Neutral/invariant behavior on uninformative or unsure answers.
    6. Sequential multi-question resolution.
    7. Explainable AI (XAI) reason generation and question bank schema integrity.
    8. FastAPI `/predict/interactive` endpoint functionality.
"""

import os
import io
import json
import unittest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.triage.image_question_engine import (
    VisualTriageQuestionEngine,
    TriageQuestion,
    QUESTION_BANK,
    CLASSES,
    NUM_CLASSES,
)
from api import app


class TestQuestionBankIntegrity(unittest.TestCase):
    """Verifies schema, dimensions, and numerical validity of the Question Bank."""

    def test_bank_contains_at_least_ten_questions(self):
        """Ensures question bank meets clinical breadth requirement (>= 10)."""
        self.assertGreaterEqual(len(QUESTION_BANK), 10)

    def test_likelihood_vectors_are_valid(self):
        """Verifies that every question has an (8,) likelihood vector bounded in [0, 1]."""
        for q_id, q in QUESTION_BANK.items():
            self.assertEqual(
                len(q.likelihood_yes),
                NUM_CLASSES,
                f"Question {q_id} likelihood dimension != {NUM_CLASSES}",
            )
            self.assertTrue(
                np.all(q.likelihood_yes >= 0.0) and np.all(q.likelihood_yes <= 1.0),
                f"Question {q_id} has likelihoods outside [0, 1]: {q.likelihood_yes}",
            )
            self.assertTrue(len(q.prompt) > 10, f"Question {q_id} prompt is too short")
            self.assertTrue(len(q.category) > 0, f"Question {q_id} category is missing")
            self.assertTrue(len(q.clinical_rationale) > 0, f"Question {q_id} rationale missing")


class TestUncertaintyAndTriggering(unittest.TestCase):
    """Verifies uncertainty calculation and trigger criteria."""

    def setUp(self):
        self.engine = VisualTriageQuestionEngine(
            confidence_threshold=0.75,
            margin_threshold=0.25,
            entropy_threshold=0.55,
        )

    def test_high_confidence_bypasses_clarification(self):
        """When top class probability >= 0.75 and margin >= 0.25, no questions are asked."""
        high_conf_probs = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.86, 0.02], dtype=np.float32)
        res = self.engine.select_clarification_questions(high_conf_probs)

        self.assertFalse(res["needs_clarification"])
        self.assertEqual(len(res["questions"]), 0)
        self.assertIn("High-confidence visual prediction", res["explanation"])

    def test_narrow_margin_triggers_clarification(self):
        """When top-1 and top-2 margin is small, active clarification is triggered."""
        ambiguous_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        ambiguous_probs[CLASSES.index("spiders")] = 0.42
        ambiguous_probs[CLASSES.index("ticks")] = 0.38
        ambiguous_probs[CLASSES.index("mosquitos")] = 0.20

        res = self.engine.select_clarification_questions(ambiguous_probs)
        self.assertTrue(res["needs_clarification"])
        self.assertGreaterEqual(len(res["questions"]), 1)
        self.assertLessEqual(len(res["questions"]), 2)
        self.assertIn("spiders", res["uncertainty_metrics"]["top1_class"])
        self.assertIn("ticks", res["uncertainty_metrics"]["top2_class"])

    def test_high_entropy_triggers_clarification(self):
        """Flat / diffuse distribution triggers clarification due to normalized entropy."""
        diffuse_probs = np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES
        res = self.engine.select_clarification_questions(diffuse_probs)

        self.assertTrue(res["needs_clarification"])
        self.assertAlmostEqual(res["uncertainty_metrics"]["entropy"], 1.0, places=2)


class TestInformationGainAndVisualBoost(unittest.TestCase):
    """Tests Mutual Information Gain and Visual Descriptor feature boosting."""

    def setUp(self):
        self.engine = VisualTriageQuestionEngine()

    def test_spider_tick_tie_selects_morphology_questions(self):
        """Ambiguous tie between spiders and ticks prioritizes necrosis or bullseye questions."""
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        probs[CLASSES.index("spiders")] = 0.45
        probs[CLASSES.index("ticks")] = 0.45
        probs[CLASSES.index("mosquitos")] = 0.10

        v_metrics = {"necrotic_center_contrast": 14.0, "lesion_circularity": 0.3}
        res = self.engine.select_clarification_questions(probs, visual_descriptors=v_metrics)

        self.assertTrue(res["needs_clarification"])
        top_q = res["questions"][0]
        # With necrotic contrast = 14.0, q_necrosis must receive a strong boost and be #1
        self.assertEqual(top_q["id"], "q_necrosis")
        self.assertGreater(top_q["visual_boost"], 1.0)
        self.assertIn("necrotic_center_contrast", top_q["visual_reason"])

    def test_circularity_boost_for_bullseye(self):
        """Elevated lesion circularity boosts q_bullseye."""
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        probs[CLASSES.index("ticks")] = 0.45
        probs[CLASSES.index("spiders")] = 0.45
        probs[CLASSES.index("no_bites")] = 0.10

        v_metrics = {"lesion_circularity": 0.85, "necrotic_center_contrast": 2.0}
        score, raw_ig, boost, v_reason = self.engine.compute_information_gain(
            probs, QUESTION_BANK["q_bullseye"], visual_descriptors=v_metrics
        )
        self.assertGreater(boost, 1.2)
        self.assertIn("lesion_circularity", v_reason)


class TestBayesianPosteriorUpdate(unittest.TestCase):
    """Verifies Bayesian belief updating on user answers."""

    def setUp(self):
        self.engine = VisualTriageQuestionEngine()

    def test_yes_answer_increases_target_posterior(self):
        """Answering Yes to necrosis markedly elevates spiders probability."""
        p_init = np.zeros(NUM_CLASSES, dtype=np.float32)
        p_init[CLASSES.index("spiders")] = 0.40
        p_init[CLASSES.index("ticks")] = 0.40
        p_init[CLASSES.index("fleas")] = 0.20

        p_updated = self.engine.update_posterior(p_init, "q_necrosis", "Yes")

        self.assertAlmostEqual(float(np.sum(p_updated)), 1.0, places=4)
        self.assertGreater(p_updated[CLASSES.index("spiders")], 0.80)
        self.assertLess(p_updated[CLASSES.index("ticks")], 0.15)

    def test_no_answer_decreases_target_posterior(self):
        """Answering No to necrosis markedly decreases spiders and boosts ticks."""
        p_init = np.zeros(NUM_CLASSES, dtype=np.float32)
        p_init[CLASSES.index("spiders")] = 0.45
        p_init[CLASSES.index("ticks")] = 0.45
        p_init[CLASSES.index("mosquitos")] = 0.10

        p_updated = self.engine.update_posterior(p_init, "q_necrosis", "No")

        self.assertAlmostEqual(float(np.sum(p_updated)), 1.0, places=4)
        self.assertLess(p_updated[CLASSES.index("spiders")], 0.15)
        self.assertGreater(p_updated[CLASSES.index("ticks")], 0.65)

    def test_not_sure_answer_is_neutral_and_invariant(self):
        """Answering 'Not sure' / 'neutral' preserves prior probabilities exactly."""
        p_init = np.array([0.05, 0.10, 0.15, 0.20, 0.20, 0.05, 0.15, 0.10], dtype=np.float32)
        p_init = p_init / np.sum(p_init)

        for ans in ["Not sure", "unsure", "skip", "neutral"]:
            p_updated = self.engine.update_posterior(p_init, "q_necrosis", ans)
            np.testing.assert_allclose(p_updated, p_init, rtol=1e-5, atol=1e-5)

    def test_sequential_question_processing(self):
        """Processes sequential answers, culminating in resolved high confidence."""
        p_init = np.zeros(NUM_CLASSES, dtype=np.float32)
        p_init[CLASSES.index("spiders")] = 0.40
        p_init[CLASSES.index("ticks")] = 0.40
        p_init[CLASSES.index("bed_bugs")] = 0.20

        answers = {"q_necrosis": "No", "q_attached_insect": "Yes"}
        p_final = self.engine.process_answered_questions(p_init, answers)

        self.assertAlmostEqual(float(np.sum(p_final)), 1.0, places=4)
        # Ticks should now dominate
        self.assertEqual(CLASSES[int(np.argmax(p_final))], "ticks")
        self.assertGreater(p_final[CLASSES.index("ticks")], 0.90)


class TestFastAPIInteractiveEndpoint(unittest.TestCase):
    """Validates FastAPI POST /predict/interactive endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def _create_dummy_image_bytes(self) -> io.BytesIO:
        img = Image.new("RGB", (224, 224), color=(180, 60, 60))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    def test_interactive_prediction_returns_valid_payload(self):
        """Ensures /predict/interactive returns questions and uncertainty metrics."""
        buf = self._create_dummy_image_bytes()
        res = self.client.post(
            "/predict/interactive",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            data={"mode": "tri_branch", "max_questions": "2"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("predicted_class", data)
        self.assertIn("confidence", data)
        self.assertIn("needs_clarification", data)
        self.assertIn("questions", data)
        self.assertIn("uncertainty_metrics", data)
        self.assertIn("visual_metrics_xai", data)

    def test_interactive_prediction_with_answers_updates_probabilities(self):
        """Verifies that providing answered_questions updates current_probabilities."""
        buf = self._create_dummy_image_bytes()
        answers = {"q_necrosis": "Yes"}

        res = self.client.post(
            "/predict/interactive",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            data={
                "mode": "tri_branch",
                "answered_questions": json.dumps(answers),
                "max_questions": "2",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["answered_questions"], answers)
        self.assertIn("current_probabilities", data)
        self.assertIn("initial_probabilities", data)
        # Spiders probability in current should be higher than initial due to Yes to necrosis
        self.assertGreater(
            data["current_probabilities"]["spiders"],
            data["initial_probabilities"]["spiders"],
        )


if __name__ == "__main__":
    unittest.main()
