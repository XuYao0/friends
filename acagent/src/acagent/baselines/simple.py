from __future__ import annotations

from acagent.runner.context import WorkflowInput
from acagent.schemas import Emotion, EmotionPrediction, Intensity


def current_only_neutral(workflow_input: WorkflowInput) -> EmotionPrediction:
    utterance = workflow_input.current_utterance
    return EmotionPrediction(
        emotions=[Emotion.NEUTRAL],
        intensities=[Intensity.NONE],
        analysis={
            "observable_facts": [f"{utterance.speaker}: {utterance.text}"],
            "memory_evidence": [],
            "inferences": [],
            "uncertainties": ["Baseline does not infer emotion."],
            "final_reason": "Current-only neutral baseline.",
        },
    )


def keyword_emotion_predictor(workflow_input: WorkflowInput) -> EmotionPrediction:
    utterance = workflow_input.current_utterance
    text = utterance.text.lower()
    if any(word in text for word in ["happy", "great", "wonderful", "love"]):
        emotion = Emotion.HAPPINESS
    elif any(word in text for word in ["angry", "mad", "hate"]):
        emotion = Emotion.ANGER
    elif any(word in text for word in ["scared", "afraid", "terrified"]):
        emotion = Emotion.FEAR
    elif any(word in text for word in ["sad", "sorry", "miss"]):
        emotion = Emotion.SADNESS
    elif any(word in text for word in ["what", "wow", "surprise"]):
        emotion = Emotion.SURPRISE
    else:
        return current_only_neutral(workflow_input)

    return EmotionPrediction(
        emotions=[emotion],
        intensities=[Intensity.LOW],
        analysis={
            "observable_facts": [f"Keyword baseline matched text: {utterance.text}"],
            "memory_evidence": [],
            "inferences": ["Emotion inferred from surface keywords only."],
            "uncertainties": ["Does not model sarcasm, context, or long-term memory."],
            "final_reason": "Surface keyword baseline.",
        },
    )
