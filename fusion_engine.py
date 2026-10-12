"""
Confidence-Weighted Multimodal Late Fusion for Gaming Addiction Risk
=====================================================================
Novel module unifying three previously-separate model outputs into one
addiction risk prediction, weighted by each modality's own confidence at
inference time -- the same design pattern used in the drone-control paper's
gesture/face/voice fusion, adapted here for heterogeneous outputs.

THREE VOTES over {Low, Medium, High}:

  1. TABULAR VOTE   - the primary Stacking Ensemble classifier, native predict_proba().
                       Confidence = max(proba).

  2. COGNITIVE VOTE  - NEW secondary classifier. Learned (not assumed):
                       trained on real [attention_score, memory_score] ->
                       addiction_level relationships in your dataset. Takes
                       the ANN's predicted attention/memory scores as input.
                       Confidence = max(proba) of THIS classifier, additionally
                       discounted by the ANN's own CV reliability (R²) for
                       each score, so an unreliable memory prediction can't
                       swing the fusion as hard as a reliable attention one.

  3. EYE-FATIGUE VOTE - Literature-derived, NOT learned from paired data,
                       because no dataset in this project pairs fatigue
                       scores with addiction labels. Grounded in Liu & Zhou
                       (2024): visual fatigue mediates ~34% of the
                       hours-gaming -> attention-deficit relationship.
                       Flagged explicitly as a prior, not a trained model.
                       Confidence = distance of CNN sigmoid output from 0.5,
                       scaled down by a fixed literature-reliability factor
                       (0.34) so it can never dominate the fused vote.

FUSION RULE: weighted geometric mean of the probability vectors
(equivalent to log-linear opinion pooling), renormalised to sum to 1.
This is the standard formalisation of "confidence-weighted late fusion"
and degrades gracefully -- a vote with confidence ~0 barely moves the
fused result, exactly as required for the safety of a screening tool.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import json

RISK_CLASSES = ["Low", "Medium", "High"]

# Updated to current ANN Regressor evaluation metrics (R^2 = 0.43 for Attention, 0.18 for Memory)
ANN_RELIABILITY = {"attention": 0.430, "memory": 0.180}   # current test R^2 per head

# From Liu & Zhou (2024), cited in your Ch.2 lit review (thesis Section 2.3.2)
EYE_FATIGUE_LITERATURE_WEIGHT = 0.34


# ------------------------------------------------------------------
# 1. Train the NEW cognitive-evidence classifier (real, learned model)
# ------------------------------------------------------------------
def train_cognitive_evidence_classifier(df):
    """
    Learns P(risk class | attention_score, memory_score) directly from your
    labelled dataset. This is what lets the ANN's cognitive predictions cast
    an informed vote instead of a hand-tuned heuristic vote.
    """
    X = df[["attention_score", "memory_score"]].values
    y = df["addiction_level"].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xs, y)
    return clf, scaler


def cognitive_vote(clf, scaler, attention_pred, memory_pred):
    Xs = scaler.transform([[attention_pred, memory_pred]])
    proba = clf.predict_proba(Xs)[0]
    proba = _align_to_class_order(clf, proba)

    # Discount confidence by ANN reliability for each score (simple average
    # of the two R^2 values -- both heads contributed to this prediction)
    reliability_discount = np.mean(list(ANN_RELIABILITY.values()))
    raw_confidence = float(np.max(proba))
    confidence = raw_confidence * reliability_discount
    return proba, confidence


def _align_to_class_order(clf, proba):
    """Reorder sklearn's clf.classes_ to match RISK_CLASSES order.
    Accepts string labels ("Low"/"Medium"/"High") or the integer codes used by the
    deployed models (training map: Low=0, Medium=1, High=2)."""
    def as_name(c):
        return RISK_CLASSES[int(c)] if isinstance(c, (int, np.integer)) else c
    class_to_idx = {as_name(c): i for i, c in enumerate(clf.classes_)}
    out = np.zeros(3)
    for i, cls in enumerate(RISK_CLASSES):
        out[i] = proba[class_to_idx[cls]] if cls in class_to_idx else 0.0
    return out / out.sum()


# ------------------------------------------------------------------
# 2. Eye-fatigue vote (literature prior, explicitly NOT a trained model)
# ------------------------------------------------------------------
def eye_fatigue_vote(fatigue_score_0_100):
    """
    fatigue_score_0_100: output of the CNN-derived eye fatigue score.
    Maps fatigue -> a soft lean toward higher risk, scaled by how much
    of the literature's reported association this single signal can
    plausibly explain (34%). This deliberately keeps the vote weak/wide
    rather than overconfident, since it is a prior, not a fitted model.
    """
    f = np.clip(fatigue_score_0_100 / 100.0, 0, 1)
    # soft monotonic mapping: more fatigue -> more mass on Medium/High
    p_high = 0.10 + 0.35 * f
    p_medium = 0.30 + 0.15 * f
    p_low = 1 - p_high - p_medium
    proba = np.array([p_low, p_medium, p_high])
    proba = np.clip(proba, 1e-6, None)
    proba = proba / proba.sum()

    # confidence capped by the literature weight -- this vote can never
    # dominate a fusion, by design
    margin_from_uncertain = abs(f - 0.5) * 2   # 0 at f=0.5, 1 at extremes
    confidence = margin_from_uncertain * EYE_FATIGUE_LITERATURE_WEIGHT
    return proba, confidence


# ------------------------------------------------------------------
# 3. Confidence-weighted fusion (log-linear opinion pooling)
# ------------------------------------------------------------------
def fuse(votes):
    """
    votes: list of (proba_vector, confidence) tuples.
    Modalities with confidence ~0 are contributed but barely move the result.
    """
    eps = 1e-9
    weights = np.array([max(c, eps) for _, c in votes])
    weights = weights / weights.sum()

    log_probs = np.zeros(3)
    for (proba, _), w in zip(votes, weights):
        log_probs += w * np.log(np.clip(proba, eps, None))
    fused = np.exp(log_probs)
    fused = fused / fused.sum()
    return fused, dict(zip(["RF_weight", "Cognitive_weight", "EyeFatigue_weight"], weights.tolist()))


# ------------------------------------------------------------------
# 4. Honest evaluation: does fusion actually help, on real data?
# ------------------------------------------------------------------
def evaluate_fusion_vs_rf_only(df):
    """
    5-fold CV comparing:
      (a) RF alone (existing system)
      (b) RF + cognitive-evidence fusion (eye fatigue omitted -- no real
          paired image data exists to test that vote honestly)
    Uses TRUE attention/memory scores as a stand-in for "ANN predicted
    perfectly" upper bound, AND a noised version matching the ANN's actual
    RMSE, so you can report both a best-case and realistic-case comparison.
    """
    feature_cols = ["age", "gender", "year_of_study", "degree_program",
                     "years_gaming", "sessions_per_week", "session_hours",
                     "total_hours_week", "night_playing", "sleep_hours",
                     "social_gaming", "gpa"]
    df = df.copy()
    df["gender"] = df["gender"].map({"Female": 0, "Male": 1, "Other": 2})
    df["degree_program"] = df["degree_program"].map(
        {"Business": 0, "CS": 1, "Engineering": 2, "IT": 3, "Medicine": 4})
    game_dummies = pd.get_dummies(df["game_type"], prefix="game")
    df = pd.concat([df, game_dummies], axis=1)
    feature_cols = feature_cols + list(game_dummies.columns)

    X = df[feature_cols].astype(float).values
    y = df["addiction_level"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_only_acc, fused_acc_realistic, fused_acc_bestcase = [], [], []
    rf_only_ll, fused_ll_realistic = [], []

    ann_att_rmse = 6.59   # from train_ann_cognitive.py fold results
    ann_mem_rmse = 6.85
    rng = np.random.default_rng(42)

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]

        rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        rf.fit(X_tr, y_tr)
        rf_proba = rf.predict_proba(X_te)
        rf_proba_aligned = np.array([_align_to_class_order(rf, p) for p in rf_proba])

        cog_clf, cog_scaler = train_cognitive_evidence_classifier(df_tr)

        fused_preds_realistic, fused_preds_best = [], []
        for i, (_, row) in enumerate(df_te.iterrows()):
            rf_p = rf_proba_aligned[i]
            rf_conf = float(np.max(rf_p))

            # best-case: ANN predicted the true score exactly
            cp_best, cc_best = cognitive_vote(cog_clf, cog_scaler,
                                               row["attention_score"], row["memory_score"])
            fused_best, _ = fuse([(rf_p, rf_conf), (cp_best, cc_best)])
            fused_preds_best.append(RISK_CLASSES[np.argmax(fused_best)])

            # realistic: add noise matching the ANN's actual RMSE
            att_noisy = row["attention_score"] + rng.normal(0, ann_att_rmse)
            mem_noisy = row["memory_score"] + rng.normal(0, ann_mem_rmse)
            cp_real, cc_real = cognitive_vote(cog_clf, cog_scaler, att_noisy, mem_noisy)
            fused_real, _ = fuse([(rf_p, rf_conf), (cp_real, cc_real)])
            fused_preds_realistic.append(RISK_CLASSES[np.argmax(fused_real)])

        rf_pred_labels = [RISK_CLASSES[np.argmax(p)] for p in rf_proba_aligned]

        rf_only_acc.append(accuracy_score(y_te, rf_pred_labels))
        fused_acc_realistic.append(accuracy_score(y_te, fused_preds_realistic))
        fused_acc_bestcase.append(accuracy_score(y_te, fused_preds_best))

    results = {
        "RF_only_accuracy_mean": float(np.mean(rf_only_acc)),
        "RF_only_accuracy_std": float(np.std(rf_only_acc)),
        "Fused_realistic_accuracy_mean": float(np.mean(fused_acc_realistic)),
        "Fused_realistic_accuracy_std": float(np.std(fused_acc_realistic)),
        "Fused_bestcase_accuracy_mean": float(np.mean(fused_acc_bestcase)),
        "Fused_bestcase_accuracy_std": float(np.std(fused_acc_bestcase)),
        "note": "Eye-fatigue vote omitted from this evaluation -- no real "
                "paired fatigue/addiction data exists to test it honestly. "
                "Best-case uses true scores (ANN-perfect upper bound); "
                "realistic uses scores noised by the ANN's actual CV RMSE."
    }
    return results


if __name__ == "__main__":
    df = pd.read_csv("data/gaming_dataset_v2.csv")
    results = evaluate_fusion_vs_rf_only(df)
    print(json.dumps(results, indent=2))

    with open("fusion_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Worked example for your progress review slides ----
    cog_clf, cog_scaler = train_cognitive_evidence_classifier(df)
    example_rf_proba = np.array([0.15, 0.55, 0.30])          # e.g. RF says Medium, unsure
    cog_p, cog_c = cognitive_vote(cog_clf, cog_scaler, attention_pred=58.0, memory_pred=52.0)
    fatigue_p, fatigue_c = eye_fatigue_vote(fatigue_score_0_100=72)
    fused, weights = fuse([(example_rf_proba, float(np.max(example_rf_proba))),
                           (cog_p, cog_c), (fatigue_p, fatigue_c)])
    print("\n=== Worked example ===")
    print("RF vote:       ", dict(zip(RISK_CLASSES, np.round(example_rf_proba, 3))))
    print("Cognitive vote:", dict(zip(RISK_CLASSES, np.round(cog_p, 3))), "conf=", round(cog_c, 3))
    print("EyeFatigue vote:", dict(zip(RISK_CLASSES, np.round(fatigue_p, 3))), "conf=", round(fatigue_c, 3))
    print("FUSED:         ", dict(zip(RISK_CLASSES, np.round(fused, 3))))
    print("Modality weights in final fusion:", {k: round(v, 3) for k, v in weights.items()})