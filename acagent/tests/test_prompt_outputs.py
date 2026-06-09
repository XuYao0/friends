import pytest

from acagent.schemas import Emotion, EmotionPrediction, Intensity


def test_emotion_prediction_validates_neutral() -> None:
    prediction = EmotionPrediction(emotions=[Emotion.NEUTRAL], intensities=[Intensity.NONE])

    prediction.validate()


def test_emotion_prediction_rejects_mixed_neutral() -> None:
    prediction = EmotionPrediction(
        emotions=[Emotion.NEUTRAL, Emotion.HAPPINESS],
        intensities=[Intensity.NONE, Intensity.LOW],
    )

    with pytest.raises(ValueError):
        prediction.validate()


def test_emotion_prediction_rejects_empty_labels() -> None:
    prediction = EmotionPrediction(emotions=[], intensities=[])

    with pytest.raises(ValueError):
        prediction.validate()
