from __future__ import annotations

from acagent.schemas import EmotionPrediction


def exact_emotion_match(prediction: EmotionPrediction, gold: EmotionPrediction) -> bool:
    return set(prediction.emotions) == set(gold.emotions) and prediction.intensities == gold.intensities

