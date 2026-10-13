"""
Confidence-Weighted Multimodal Late Fusion for Gaming Addiction Risk
=====================================================================
Combines three model outputs into one risk prediction over {Low, Medium, High},
weighted by each modality's own confidence at inference time.

THREE VOTES:

  1. BEHAVIOUR VOTE  - the real-data IGD model (notebooks/07_train_real_igd_model.py),
                       trained on 322 Sri Lankan students with the IGDS9-SF
                       (Manchanayake, 2021). It predicts the IGDS9-SF score; tier
                       probabilities come from that score and the model's
                       cross-validated error. Confidence = max(proba).

  2. COGNITIVE VOTE  - literature prior from the measured Go/No-Go task.
                       No public dataset pairs these cognitive tests with
                       IGDS9-SF scores, so this vote is NOT a trained model.
                       Impaired response inhibition is the best-supported
                       cognitive marker of IGD: meta-analytic d = 0.56
                       (Argyriou, Davison & Lee, 2017), i.e. r ≈ 0.27, which
                       caps this vote's confidence.

  3. EYE-FATIGUE VOTE - literature prior (no dataset pairs eye fatigue with IGD
                       labels). Weight from Liu & Zhou (2024): visual fatigue
                       mediates ~34% of the gaming-hours -> attention-deficit link.

FUSION RULE: weighted geometric mean of the probability vectors (log-linear
opinion pooling), renormalised. A vote with confidence ~0 barely moves the
result. The fusion's benefit over the behaviour model alone can only be
evaluated on real paired data, i.e. the Validation Study (analyze_study.py).
"""
import numpy as np

RISK_CLASSES = ["Low", "Medium", "High"]

# Liu & Zhou (2024): fatigue explains ~34% of the hours -> attention-loss relationship
EYE_FATIGUE_LITERATURE_WEIGHT = 0.34

# Argyriou, Davison & Lee (2017): d = 0.56 for impaired response inhibition in IGD.
# Converted to a correlation, r = d / sqrt(d^2 + 4) ≈ 0.27
INHIBITION_EFFECT_D = 0.56
COGNITIVE_LITERATURE_WEIGHT = INHIBITION_EFFECT_D / np.sqrt(INHIBITION_EFFECT_D ** 2 + 4)


def _soft_lean(f):
    """Monotonic lean toward higher risk as f goes 0 -> 1 (shared by both prior votes)."""
    p_high = 0.10 + 0.35 * f
    p_medium = 0.30 + 0.15 * f
    proba = np.clip(np.array([1 - p_high - p_medium, p_medium, p_high]), 1e-6, None)
    return proba / proba.sum()


# ------------------------------------------------------------------
# 1. Behaviour vote (real-data IGDS9-SF model)
# ------------------------------------------------------------------
def behaviour_vote(tier_proba):
    proba = np.asarray(tier_proba, dtype=float)
    proba = np.clip(proba, 1e-6, None)
    proba = proba / proba.sum()
    return proba, float(np.max(proba))


# ------------------------------------------------------------------
# 2. Cognitive vote (literature prior from Go/No-Go response inhibition)
# ------------------------------------------------------------------
def cognitive_inhibition_vote(commission_rate_pct):
    """
    commission_rate_pct: % of No-Go trials where the student pressed anyway (0-100).
    More commission errors -> weaker response inhibition -> lean toward higher risk.
    """
    f = float(np.clip(commission_rate_pct / 100.0, 0, 1))
    margin_from_uncertain = abs(f - 0.5) * 2
    return _soft_lean(f), margin_from_uncertain * COGNITIVE_LITERATURE_WEIGHT


# ------------------------------------------------------------------
# 3. Eye-fatigue vote (literature prior, explicitly NOT a trained model)
# ------------------------------------------------------------------
def eye_fatigue_vote(fatigue_score_0_100):
    """fatigue_score_0_100: ResNet50 + OpenCV eye fatigue score from the app."""
    f = float(np.clip(fatigue_score_0_100 / 100.0, 0, 1))
    margin_from_uncertain = abs(f - 0.5) * 2   # 0 at f=0.5, 1 at extremes
    return _soft_lean(f), margin_from_uncertain * EYE_FATIGUE_LITERATURE_WEIGHT


# ------------------------------------------------------------------
# 4. Confidence-weighted fusion (log-linear opinion pooling)
# ------------------------------------------------------------------
def fuse(votes):
    """
    votes: list of (proba_vector, confidence) tuples, in the order
    behaviour, cognitive, eye fatigue. A missing modality can be passed
    with confidence 0 and contributes (almost) nothing.
    """
    eps = 1e-9
    weights = np.array([max(c, eps) for _, c in votes])
    weights = weights / weights.sum()

    log_probs = np.zeros(3)
    for (proba, _), w in zip(votes, weights):
        log_probs += w * np.log(np.clip(proba, eps, None))
    fused = np.exp(log_probs)
    fused = fused / fused.sum()
    return fused, dict(zip(["Behaviour_weight", "Cognitive_weight", "EyeFatigue_weight"], weights.tolist()))


if __name__ == "__main__":
    # Worked example: behaviour model unsure between Medium and High, weak inhibition, tired eyes
    behaviour_p, behaviour_c = behaviour_vote([0.10, 0.55, 0.35])
    cog_p, cog_c = cognitive_inhibition_vote(60)
    eye_p, eye_c = eye_fatigue_vote(72)
    fused, weights = fuse([(behaviour_p, behaviour_c), (cog_p, cog_c), (eye_p, eye_c)])
    print("Behaviour vote:", dict(zip(RISK_CLASSES, np.round(behaviour_p, 3))), "conf", round(behaviour_c, 3))
    print("Cognitive vote:", dict(zip(RISK_CLASSES, np.round(cog_p, 3))), "conf", round(cog_c, 3))
    print("Eye vote:      ", dict(zip(RISK_CLASSES, np.round(eye_p, 3))), "conf", round(eye_c, 3))
    print("FUSED:         ", dict(zip(RISK_CLASSES, np.round(fused, 3))))
    print("Weights:       ", {k: round(v, 3) for k, v in weights.items()})
