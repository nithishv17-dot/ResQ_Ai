"""
src.triage.image_question_engine
================================
Image-Based Dynamic Questioning & Visual-Triage Active Clarification Engine.

Provides:
- VisualTriageQuestionEngine: Active learning / clarification engine that assesses
  ensemble prediction uncertainty, computes information gain (mutual information)
  over candidate questions conditioned on visual descriptors, and executes
  Bayesian posterior updates when users provide answers.
- QUESTION_BANK: Curated clinical and environmental discriminative questions with
  rigorous likelihood models P(Answer | Class) across the 8 insect bite classes.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("VisualTriageQuestionEngine")

CLASSES = [
    "ants",
    "bed_bugs",
    "chiggers",
    "fleas",
    "mosquitos",
    "no_bites",
    "spiders",
    "ticks",
]
NUM_CLASSES = len(CLASSES)
EPSILON = 1e-12
LOG_NUM_CLASSES = math.log(NUM_CLASSES)


@dataclass
class TriageQuestion:
    """Representation of a discriminative visual-triage question."""

    id: str
    prompt: str
    category: str
    clinical_rationale: str
    # Likelihood array P(Yes | class_i) for i in [0..7]
    likelihood_yes: np.ndarray
    visual_cue_metric: Optional[str] = None
    visual_cue_threshold: float = 0.0
    visual_cue_boost_weight: float = 1.0


# ---------------------------------------------------------------------------
# Structured Discriminative Question Bank (11 Clinical & Environmental Items)
# ---------------------------------------------------------------------------
# Order: ['ants', 'bed_bugs', 'chiggers', 'fleas', 'mosquitos', 'no_bites', 'spiders', 'ticks']
QUESTION_BANK: Dict[str, TriageQuestion] = {
    "q_necrosis": TriageQuestion(
        id="q_necrosis",
        prompt="Is there a dark, purple, or sunken black center/spot in the middle of the bite?",
        category="Lesion Morphology",
        clinical_rationale="Key hallmark of necrotic arachnid venom (loxoscelism / spider bite) vs superficial insect bites.",
        likelihood_yes=np.array([0.15, 0.04, 0.04, 0.04, 0.04, 0.01, 0.88, 0.08], dtype=np.float32),
        visual_cue_metric="necrotic_center_contrast",
        visual_cue_threshold=8.0,
        visual_cue_boost_weight=1.5,
    ),
    "q_bullseye": TriageQuestion(
        id="q_bullseye",
        prompt="Is there an expanding circular or bullseye ring (red outer ring with clearing around center)?",
        category="Lesion Morphology",
        clinical_rationale="Hallmark pathognomonic sign of Erythema Migrans from Borrelia burgdorferi tick attachment.",
        likelihood_yes=np.array([0.04, 0.05, 0.05, 0.05, 0.08, 0.01, 0.12, 0.92], dtype=np.float32),
        visual_cue_metric="lesion_circularity",
        visual_cue_threshold=0.55,
        visual_cue_boost_weight=1.5,
    ),
    "q_attached_insect": TriageQuestion(
        id="q_attached_insect",
        prompt="Did you find an insect or tick firmly attached or embedded in your skin?",
        category="Physical Evidence",
        clinical_rationale="Ticks frequently remain embedded for 24-72 hours, unlike transient stingers/biters.",
        likelihood_yes=np.array([0.05, 0.03, 0.08, 0.05, 0.03, 0.01, 0.03, 0.90], dtype=np.float32),
    ),
    "q_linear_cluster": TriageQuestion(
        id="q_linear_cluster",
        prompt="Are the bites arranged in a straight line or clustered in groups of 3 or more ('breakfast, lunch, and dinner')?",
        category="Lesion Distribution",
        clinical_rationale="Distinguishes clustered feeders (bed bugs, fleas) from solitary bites (spiders, ticks).",
        likelihood_yes=np.array([0.20, 0.88, 0.35, 0.72, 0.20, 0.02, 0.05, 0.05], dtype=np.float32),
        visual_cue_metric="lesion_eccentricity",
        visual_cue_threshold=0.70,
        visual_cue_boost_weight=1.3,
    ),
    "q_pustule": TriageQuestion(
        id="q_pustule",
        prompt="Has a small fluid-filled blister or white/yellow pustule formed at the center of the bump?",
        category="Lesion Morphology",
        clinical_rationale="Pathognomonic of Solenopsis (fire ant) venom within 24 hours of sting.",
        likelihood_yes=np.array([0.90, 0.05, 0.20, 0.15, 0.05, 0.01, 0.30, 0.05], dtype=np.float32),
        visual_cue_metric="texture_contrast",
        visual_cue_threshold=0.90,
        visual_cue_boost_weight=1.4,
    ),
    "q_intense_itching": TriageQuestion(
        id="q_intense_itching",
        prompt="Is the area experiencing intense, relentless itching rather than sharp or throbbing pain?",
        category="Symptomatology",
        clinical_rationale="Pruritus is dominant in chiggers, fleas, and mosquitos; spiders and ants produce sharp pain/burning.",
        likelihood_yes=np.array([0.35, 0.85, 0.96, 0.92, 0.88, 0.05, 0.15, 0.20], dtype=np.float32),
        visual_cue_metric="erythema_score",
        visual_cue_threshold=0.45,
        visual_cue_boost_weight=1.2,
    ),
    "q_severe_pain": TriageQuestion(
        id="q_severe_pain",
        prompt="Did you feel an immediate sharp stinging or severe burning/throbbing pain at the moment of the bite?",
        category="Symptomatology",
        clinical_rationale="Ants and spiders cause immediate pain/burning; ticks and bed bugs inject painless anesthetics.",
        likelihood_yes=np.array([0.92, 0.04, 0.08, 0.15, 0.15, 0.01, 0.80, 0.05], dtype=np.float32),
        visual_cue_metric="erythema_score",
        visual_cue_threshold=0.50,
        visual_cue_boost_weight=1.2,
    ),
    "q_bed_exposure": TriageQuestion(
        id="q_bed_exposure",
        prompt="Did you wake up with these new bites after sleeping in a bed, hotel, or upholstered furniture?",
        category="Environmental Exposure",
        clinical_rationale="Distinguishes nocturnal Cimex lectularius (bed bug) infestations from outdoor vector bites.",
        likelihood_yes=np.array([0.08, 0.92, 0.05, 0.30, 0.35, 0.05, 0.20, 0.05], dtype=np.float32),
    ),
    "q_pet_exposure": TriageQuestion(
        id="q_pet_exposure",
        prompt="Have you been in close contact with dogs, cats, pet bedding, or carpeted areas recently?",
        category="Environmental Exposure",
        clinical_rationale="Primary vector reservoir for Ctenocephalides felis (flea) infestations.",
        likelihood_yes=np.array([0.15, 0.15, 0.15, 0.90, 0.25, 0.10, 0.15, 0.35], dtype=np.float32),
    ),
    "q_outdoors_tall_grass": TriageQuestion(
        id="q_outdoors_tall_grass",
        prompt="Were you walking in tall grass, woods, brush, or forest trails recently?",
        category="Environmental Exposure",
        clinical_rationale="Primary habitat for Ixodidae ticks and Trombiculidae chigger mites.",
        likelihood_yes=np.array([0.40, 0.05, 0.90, 0.30, 0.65, 0.10, 0.40, 0.92], dtype=np.float32),
    ),
    "q_tight_clothing_location": TriageQuestion(
        id="q_tight_clothing_location",
        prompt="Are the bites specifically concentrated around tight clothing bands (waistband, sock line, groin, armpits)?",
        category="Lesion Distribution",
        clinical_rationale="Classic anatomic predilection of chigger stylostome feeding sites.",
        likelihood_yes=np.array([0.25, 0.15, 0.94, 0.45, 0.10, 0.02, 0.15, 0.45], dtype=np.float32),
    ),
}


class VisualTriageQuestionEngine:
    """
    Active clarification engine for ambiguous insect bite classifications.

    Evaluates ensemble prediction uncertainty, calculates mutual information gain
    conditioned on visual descriptors, and performs Bayesian posterior updates
    upon user responses.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        margin_threshold: float = 0.25,
        entropy_threshold: float = 0.55,
        question_bank: Optional[Dict[str, TriageQuestion]] = None,
    ):
        """
        Args:
            confidence_threshold: If Top-1 confidence < threshold, triggers clarification.
            margin_threshold: If Top-1 prob - Top-2 prob < threshold, triggers clarification.
            entropy_threshold: If normalized Shannon entropy > threshold, triggers clarification.
            question_bank: Optional custom question bank mapping ID -> TriageQuestion.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.question_bank = question_bank or QUESTION_BANK

    @staticmethod
    def calculate_normalized_entropy(probs: np.ndarray) -> float:
        """
        Calculates normalized Shannon entropy H(P) in range [0, 1].

        H(P) = - sum(p_i * ln(p_i)) / ln(K)
        """
        p = np.clip(np.asarray(probs, dtype=np.float32), EPSILON, 1.0)
        p = p / np.sum(p)
        raw_entropy = -float(np.sum(p * np.log(p)))
        norm_entropy = raw_entropy / LOG_NUM_CLASSES
        return float(np.clip(norm_entropy, 0.0, 1.0))

    def evaluate_uncertainty(self, vision_probs: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates uncertainty metrics for a given vision probability vector.

        Returns:
            Dictionary with:
                - top1_idx, top1_class, top1_prob
                - top2_idx, top2_class, top2_prob
                - margin (top1_prob - top2_prob)
                - entropy (normalized Shannon entropy)
                - needs_clarification (bool)
                - trigger_reasons (List[str])
        """
        p = np.asarray(vision_probs, dtype=np.float32).flatten()
        if len(p) != NUM_CLASSES:
            raise ValueError(f"Expected {NUM_CLASSES} class probabilities, got {len(p)}")

        p = np.clip(p, EPSILON, 1.0)
        p = p / np.sum(p)

        sorted_indices = np.argsort(p)[::-1]
        top1_idx = int(sorted_indices[0])
        top2_idx = int(sorted_indices[1])

        top1_prob = float(p[top1_idx])
        top2_prob = float(p[top2_idx])
        margin = float(top1_prob - top2_prob)
        entropy = self.calculate_normalized_entropy(p)

        trigger_reasons = []
        if top1_prob < self.confidence_threshold:
            trigger_reasons.append(
                f"Top-1 confidence ({top1_prob*100:.1f}%) < threshold ({self.confidence_threshold*100:.1f}%)"
            )
        if margin < self.margin_threshold:
            trigger_reasons.append(
                f"Candidate margin between '{CLASSES[top1_idx]}' and '{CLASSES[top2_idx]}' ({margin*100:.1f}%) < threshold ({self.margin_threshold*100:.1f}%)"
            )
        if entropy > self.entropy_threshold:
            trigger_reasons.append(
                f"Normalized Shannon entropy ({entropy:.3f}) > threshold ({self.entropy_threshold:.3f})"
            )

        needs_clarification = len(trigger_reasons) > 0

        return {
            "top1_idx": top1_idx,
            "top1_class": CLASSES[top1_idx],
            "top1_prob": top1_prob,
            "top2_idx": top2_idx,
            "top2_class": CLASSES[top2_idx],
            "top2_prob": top2_prob,
            "margin": margin,
            "entropy": entropy,
            "needs_clarification": needs_clarification,
            "trigger_reasons": trigger_reasons,
        }

    def compute_information_gain(
        self,
        current_probs: np.ndarray,
        question: TriageQuestion,
        visual_descriptors: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float, str]:
        """
        Computes expected mutual information gain IG(Q) = H(C) - H(C | Q)
        and applies visual descriptor feature boosts.

        Returns:
            (composite_score, raw_ig, visual_boost, visual_reason)
        """
        p = np.clip(np.asarray(current_probs, dtype=np.float32), EPSILON, 1.0)
        p = p / np.sum(p)

        # Baseline entropy H(C)
        h_current = self.calculate_normalized_entropy(p)

        p_yes_given_c = question.likelihood_yes
        p_no_given_c = 1.0 - p_yes_given_c

        # Marginal probabilities of answers
        p_yes = float(np.sum(p_yes_given_c * p))
        p_no = float(1.0 - p_yes)

        # Numerical safety
        p_yes = max(EPSILON, min(1.0 - EPSILON, p_yes))
        p_no = max(EPSILON, min(1.0 - EPSILON, p_no))

        # Posterior distributions given Yes / No
        post_yes = (p_yes_given_c * p) / p_yes
        post_no = (p_no_given_c * p) / p_no

        h_yes = self.calculate_normalized_entropy(post_yes)
        h_no = self.calculate_normalized_entropy(post_no)

        expected_conditional_entropy = (p_yes * h_yes) + (p_no * h_no)
        raw_ig = max(0.0, float(h_current - expected_conditional_entropy))

        # Visual descriptor boost
        visual_boost = 1.0
        visual_reason = ""
        vdesc = visual_descriptors or {}

        if question.visual_cue_metric and question.visual_cue_metric in vdesc:
            val = float(vdesc.get(question.visual_cue_metric, 0.0))
            thresh = question.visual_cue_threshold
            if val > thresh:
                # Dynamic scaling proportional to threshold exceedance
                excess = val - thresh
                scale = min(1.5, excess / (thresh + 1e-6))
                visual_boost = 1.0 + (scale * question.visual_cue_boost_weight)
                visual_reason = (
                    f"Elevated visual biomarker '{question.visual_cue_metric}' = {val:.2f} (threshold: {thresh:.2f}) "
                    f"boosted prioritization x{visual_boost:.2f}."
                )

        composite_score = raw_ig * visual_boost
        return float(composite_score), float(raw_ig), float(visual_boost), visual_reason

    def _generate_xai_rationale(
        self,
        question: TriageQuestion,
        unc: Dict[str, Any],
        raw_ig: float,
        visual_boost: float,
        visual_reason: str,
    ) -> str:
        """Constructs an explainable medical reasoning string for the selected question."""
        c1 = unc["top1_class"]
        c2 = unc["top2_class"]
        p1 = unc["top1_prob"] * 100
        p2 = unc["top2_prob"] * 100

        c1_idx = unc["top1_idx"]
        c2_idx = unc["top2_idx"]
        p_yes_c1 = question.likelihood_yes[c1_idx]
        p_yes_c2 = question.likelihood_yes[c2_idx]

        # Determine which class is favored by 'Yes' vs 'No'
        if p_yes_c1 > p_yes_c2:
            favor_cls = c1
            other_cls = c2
            f_prob = p_yes_c1
            o_prob = p_yes_c2
        else:
            favor_cls = c2
            other_cls = c1
            f_prob = p_yes_c2
            o_prob = p_yes_c1

        explanation = (
            f"Differentiates top ambiguous candidates '{c1}' ({p1:.1f}%) vs '{c2}' ({p2:.1f}%). "
            f"Answering 'Yes' provides strong diagnostic evidence favoring '{favor_cls}' (P={f_prob:.2f}) "
            f"over '{other_cls}' (P={o_prob:.2f}) with expected Information Gain = {raw_ig:.3f} bits."
        )
        if visual_reason:
            explanation += f" [Visual Biomarker: {visual_reason}]"

        return explanation

    def select_clarification_questions(
        self,
        vision_probs: np.ndarray,
        visual_descriptors: Optional[Dict[str, float]] = None,
        asked_questions: Optional[List[str]] = None,
        max_questions: int = 2,
    ) -> Dict[str, Any]:
        """
        Determines if clarification is needed and ranks available questions by Information Gain.

        Args:
            vision_probs: 8-class probability array.
            visual_descriptors: Dictionary of extracted visual metrics.
            asked_questions: List of question IDs that have already been answered.
            max_questions: Maximum number of questions to recommend (default: 2).

        Returns:
            Dictionary containing:
                - needs_clarification (bool)
                - uncertainty_metrics (dict)
                - questions (List[dict])
                - explanation (str)
        """
        unc = self.evaluate_uncertainty(vision_probs)
        asked = set(asked_questions or [])

        if not unc["needs_clarification"]:
            return {
                "needs_clarification": False,
                "uncertainty_metrics": unc,
                "questions": [],
                "explanation": (
                    f"High-confidence visual prediction for '{unc['top1_class']}' "
                    f"({unc['top1_prob']*100:.1f}%, margin: {unc['margin']*100:.1f}%, entropy: {unc['entropy']:.3f}). "
                    f"No follow-up questions required."
                ),
            }

        # Rank unasked questions by information gain * visual boost
        scored_questions = []
        for q_id, question in self.question_bank.items():
            if q_id in asked:
                continue

            score, raw_ig, boost, v_reason = self.compute_information_gain(
                vision_probs, question, visual_descriptors
            )
            xai_reason = self._generate_xai_rationale(question, unc, raw_ig, boost, v_reason)

            scored_questions.append({
                "id": question.id,
                "prompt": question.prompt,
                "category": question.category,
                "clinical_rationale": question.clinical_rationale,
                "information_gain": round(float(raw_ig), 4),
                "visual_boost": round(float(boost), 2),
                "composite_score": round(float(score), 4),
                "visual_reason": v_reason,
                "xai_explanation": xai_reason,
                "options": ["Yes", "No", "Not sure"],
            })

        # Sort descending by composite score
        scored_questions.sort(key=lambda x: x["composite_score"], reverse=True)
        selected = scored_questions[:max_questions]

        summary_explanation = (
            f"Active triage clarification triggered: {'; '.join(unc['trigger_reasons'])}. "
            f"Identified {len(selected)} high-information discriminative questions."
        )

        return {
            "needs_clarification": len(selected) > 0,
            "uncertainty_metrics": unc,
            "questions": selected,
            "explanation": summary_explanation,
        }

    def update_posterior(
        self,
        current_probs: np.ndarray,
        question_id: str,
        answer: str,
    ) -> np.ndarray:
        """
        Executes a Bayesian posterior probability update given an answer to a question.

        P(Class | Answer) proportional to P(Answer | Class) * P(Class)

        Args:
            current_probs: Prior probability distribution across the 8 classes.
            question_id: Identifier of the answered question.
            answer: Spoken or selected answer ('Yes', 'No', or 'Not sure').

        Returns:
            Updated (8,) normalized posterior probability vector.
        """
        p = np.clip(np.asarray(current_probs, dtype=np.float32).flatten(), EPSILON, 1.0)
        p = p / np.sum(p)

        if question_id not in self.question_bank:
            logger.warning(f"Question ID '{question_id}' not found in question bank. Returning unchanged probs.")
            return p

        question = self.question_bank[question_id]
        clean_ans = str(answer).strip().lower()

        if clean_ans in ["yes", "y", "true", "positive"]:
            likelihood = question.likelihood_yes
        elif clean_ans in ["no", "n", "false", "negative"]:
            likelihood = 1.0 - question.likelihood_yes
        elif clean_ans in ["not sure", "unsure", "skip", "neutral", "unknown"]:
            # Neutral / uninformative observation leaves posterior equal to prior
            likelihood = np.ones(NUM_CLASSES, dtype=np.float32)
        else:
            logger.warning(f"Unrecognized answer '{answer}' for '{question_id}'. Treating as neutral.")
            likelihood = np.ones(NUM_CLASSES, dtype=np.float32)

        # Bayesian multiplication
        unnormalized_posterior = p * likelihood
        evidence = float(np.sum(unnormalized_posterior))

        if evidence < EPSILON:
            logger.warning("Posterior evidence is zero; returning prior.")
            return p

        posterior = unnormalized_posterior / evidence
        return np.asarray(posterior, dtype=np.float32)

    def process_answered_questions(
        self,
        initial_probs: np.ndarray,
        answers: Dict[str, str],
    ) -> np.ndarray:
        """
        Sequentially updates the probability distribution across a dict of answered questions.

        Args:
            initial_probs: Starting vision probability array.
            answers: Dict mapping question_id -> answer ('Yes'/'No'/'Not sure').

        Returns:
            Final updated posterior probability array (8,).
        """
        probs = np.asarray(initial_probs, dtype=np.float32).copy()
        for q_id, ans in answers.items():
            probs = self.update_posterior(probs, q_id, ans)
        return probs
