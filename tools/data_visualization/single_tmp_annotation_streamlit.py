#!/usr/bin/env python3
"""Single tmp.json Streamlit annotation UI."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


DEFAULT_TMP_PATH = Path("workzone/tmp.json")
DEFAULT_LABELS_PATH = Path("workzone/single_tmp_xy_labels.json")

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "surprise",
    "sadness",
    "contentment",
    "relief",
    "interest",
    "contempt",
    "shame",
    "guilt",
    "embarrassment",
    "neutral",
]
INTENSITIES = ["low", "medium", "high", "none"]


def read_json_file(path: Path, attempts: int = 5, delay: float = 0.08) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read {path}")


def load_tmp(path: Path) -> dict[str, Any]:
    try:
        data = read_json_file(path)
    except FileNotFoundError:
        st.error(f"Not found: {path}")
        st.stop()
    except json.JSONDecodeError as exc:
        cached = st.session_state.get(f"last_tmp_data:{path}")
        if cached is not None:
            st.warning(f"{path} is being written right now; showing last loaded snapshot.")
            return cached
        st.error(f"Could not parse {path}: {exc}")
        st.stop()

    if not isinstance(data.get("results"), list):
        st.error(f"{path} does not contain a results list.")
        st.stop()
    st.session_state[f"last_tmp_data:{path}"] = data
    return data


def load_manual_labels(path: Path, source_path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "metadata": {
                "source": str(source_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "labels": {},
        }
    try:
        data = read_json_file(path)
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse {path}: {exc}")
        st.stop()
    if not isinstance(data, dict):
        st.error(f"{path} must contain a JSON object.")
        st.stop()
    labels = data.setdefault("labels", {})
    if not isinstance(labels, dict):
        st.error(f"{path} field labels must be a JSON object.")
        st.stop()
    data.setdefault("metadata", {})
    return data


def save_manual_label(
    path: Path,
    source_path: Path,
    utterance_key: str,
    emotions: list[str],
    intensities: list[str],
    reason: str,
) -> None:
    data = load_manual_labels(path, source_path)
    data["metadata"]["source"] = str(source_path)
    data["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    first_emotion = emotions[0] if emotions else "neutral"
    first_intensity = intensities[0] if intensities else "none"
    data["labels"][utterance_key] = {
        "xy_label": {
            "emotions": emotions,
            "intensities": intensities,
            "emotion": first_emotion,
            "intensity": first_intensity,
            "reason": reason.strip(),
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def result_utterance(result: dict[str, Any]) -> dict[str, Any]:
    utterance = result.get("utterance")
    return utterance if isinstance(utterance, dict) else {}


def result_utterance_id(result: dict[str, Any]) -> int | None:
    utterance_id = result_utterance(result).get("utterance_id")
    return utterance_id if isinstance(utterance_id, int) else None


def result_global_utterance_id(result: dict[str, Any]) -> int | None:
    global_id = result_utterance(result).get("global_utterance_id")
    return global_id if isinstance(global_id, int) else None


def result_key(result: dict[str, Any]) -> str:
    global_id = result_global_utterance_id(result)
    if global_id is not None:
        return f"global:{global_id}"
    utterance_id = result_utterance_id(result)
    if utterance_id is not None:
        return f"local:{utterance_id}"
    return f"row:{id(result)}"


def label_pairs_html(emotions: Any, intensities: Any) -> str:
    chips = []
    if isinstance(emotions, list) and isinstance(intensities, list):
        for emotion, intensity in zip(emotions, intensities):
            chips.append(f'<span class="label-chip">{emotion} / {intensity}</span>')
    return " ".join(chips) if chips else '<span class="label-chip">(missing)</span>'


def xy_emotions(xy_label: dict[str, Any]) -> list[str]:
    emotions = xy_label.get("emotions")
    if isinstance(emotions, list):
        return [emotion for emotion in emotions if emotion in EMOTIONS]
    emotion = xy_label.get("emotion")
    return [emotion] if emotion in EMOTIONS else []


def xy_intensities(xy_label: dict[str, Any], emotions: list[str]) -> list[str]:
    intensities = xy_label.get("intensities")
    if isinstance(intensities, list) and len(intensities) == len(emotions):
        return [intensity if intensity in INTENSITIES else "medium" for intensity in intensities]
    intensity = xy_label.get("intensity")
    if intensity in INTENSITIES and emotions:
        return [intensity] + ["medium"] * (len(emotions) - 1)
    return ["none" if emotion == "neutral" else "medium" for emotion in emotions]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="stApp"] {
            font-size: 18px;
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            font-size: 1.05rem;
            line-height: 1.55;
        }
        textarea {
            font-size: 1.05rem !important;
            line-height: 1.55 !important;
        }
        button {
            font-size: 1.02rem !important;
        }
        .info-block {
            border-left: 5px solid;
            border-radius: 6px;
            padding: 0.85rem 1rem;
            margin: 0.4rem 0 1rem 0;
        }
        .utterance-block {
            background: #f7fbff;
            border-color: #2563eb;
        }
        .model-block {
            background: #fff8eb;
            border-color: #d97706;
        }
        .xy-block {
            background: #f0fdf4;
            border-color: #16a34a;
        }
        .block-title {
            color: #374151;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }
        .speaker {
            color: #1d4ed8;
            font-weight: 800;
        }
        .utterance-text {
            color: #111827;
            font-size: 1.28rem;
            line-height: 1.55;
        }
        .label-chip {
            display: inline-block;
            background: #1f2937;
            color: white;
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            margin: 0 0.25rem 0.25rem 0;
            font-size: 1rem;
            font-weight: 650;
        }
        .reason-text {
            color: #374151;
            font-size: 1.08rem;
            line-height: 1.58;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_current_index(index: int, max_index: int) -> None:
    st.session_state["single_tmp_current_index"] = min(max(index, 0), max_index)


def main() -> None:
    st.set_page_config(page_title="Single TMP Annotation", layout="wide")
    inject_css()
    st.title("Single TMP Annotation")

    with st.sidebar:
        tmp_path = Path(st.text_input("tmp.json path", value=str(DEFAULT_TMP_PATH)))
        labels_path = Path(st.text_input("manual labels path", value=str(DEFAULT_LABELS_PATH)))
        st.caption("Manual labels are written to the overlay file only.")

    tmp_data = load_tmp(tmp_path)
    manual_data = load_manual_labels(labels_path, tmp_path)
    results = [item for item in tmp_data["results"] if isinstance(item, dict)]
    if not results:
        st.info("No results loaded.")
        st.stop()

    max_index = len(results) - 1
    if "single_tmp_current_index" not in st.session_state:
        st.session_state["single_tmp_current_index"] = 0
    current_index = min(st.session_state["single_tmp_current_index"], max_index)
    st.session_state["single_tmp_current_index"] = current_index

    left, middle, right, jump = st.columns([1, 1, 1, 3])
    with left:
        if st.button("上一个", disabled=current_index == 0, use_container_width=True):
            set_current_index(current_index - 1, max_index)
            st.rerun()
    with middle:
        st.metric("当前序号", f"{current_index + 1}/{len(results)}")
    with right:
        if st.button("下一个", disabled=current_index == max_index, use_container_width=True):
            set_current_index(current_index + 1, max_index)
            st.rerun()
    with jump:
        target_number = st.number_input(
            "指定序号跳转",
            min_value=1,
            max_value=len(results),
            value=current_index + 1,
            step=1,
        )
        if st.button("跳转", use_container_width=True):
            set_current_index(int(target_number) - 1, max_index)
            st.rerun()

    st.caption(
        f"Loaded {len(results)} results from {tmp_path}; manual labels: "
        f"{len(manual_data.get('labels', {}))}; saved to {labels_path}."
    )

    result = results[current_index]
    utterance = result_utterance(result)
    key = result_key(result)
    label = result.get("label", {})
    model_label = label if isinstance(label, dict) else {}
    xy_entry = manual_data["labels"].get(key, {})
    xy_label = xy_entry.get("xy_label", {}) if isinstance(xy_entry, dict) else {}

    speaker = utterance.get("speaker", "")
    text = utterance.get("utterance", "")
    inline_description = utterance.get("inline_description") or []
    inline_html = ""
    if inline_description:
        inline_html = f'<div class="reason-text">{"; ".join(str(item) for item in inline_description)}</div>'
    st.markdown(
        f"""
        <div class="info-block utterance-block">
          <div class="block-title">Utterance</div>
          <div class="utterance-text"><span class="speaker">{speaker}</span>: {text}</div>
          {inline_html}
          <div class="reason-text">key={key}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-block model-block">
          <div class="block-title">Model Label</div>
          <div>{label_pairs_html(model_label.get("emotions"), model_label.get("intensities"))}</div>
          <div class="reason-text">{model_label.get("reason", "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if xy_label:
        current_xy_emotions = xy_emotions(xy_label)
        current_xy_intensities = xy_intensities(xy_label, current_xy_emotions)
        st.markdown(
            f"""
            <div class="info-block xy-block">
              <div class="block-title">Manual Label</div>
              <div>{label_pairs_html(current_xy_emotions, current_xy_intensities)}</div>
              <div class="reason-text">{xy_label.get("reason", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Manual Annotation")
    default_emotions = xy_emotions(xy_label)
    if not default_emotions and isinstance(model_label.get("emotions"), list):
        default_emotions = [emotion for emotion in model_label["emotions"] if emotion in EMOTIONS]
    if not default_emotions:
        default_emotions = ["neutral"]
    default_intensities = xy_intensities(xy_label, default_emotions)

    selected_emotions = st.multiselect(
        "Emotions",
        EMOTIONS,
        default=default_emotions,
        key=f"single_emotions_{key}",
    )
    if "neutral" in selected_emotions and len(selected_emotions) > 1:
        st.warning("neutral 不能和其他 emotion 混选；提交时会要求你只保留一种选择。")

    selected_intensities: list[str] = []
    if selected_emotions:
        st.write("Intensity")
        intensity_cols = st.columns(min(3, len(selected_emotions)))
        default_intensity_by_emotion = dict(zip(default_emotions, default_intensities))
        for index, emotion in enumerate(selected_emotions):
            options = ["none"] if emotion == "neutral" else ["low", "medium", "high"]
            default_value = default_intensity_by_emotion.get(
                emotion,
                "none" if emotion == "neutral" else "medium",
            )
            if default_value not in options:
                default_value = "none" if emotion == "neutral" else "medium"
            with intensity_cols[index % len(intensity_cols)]:
                selected_intensities.append(
                    st.radio(
                        emotion,
                        options,
                        index=options.index(default_value),
                        horizontal=True,
                        key=f"single_intensity_{key}_{emotion}",
                    )
                )

    reason = st.text_area(
        "Reason",
        value=xy_label.get("reason") or model_label.get("reason", ""),
        height=100,
        key=f"single_reason_{key}",
    )

    if st.button("提交标注", type="primary"):
        if not reason.strip():
            st.error("Reason 不能为空。")
        elif not selected_emotions:
            st.error("至少选择一个 emotion。")
        elif "neutral" in selected_emotions and len(selected_emotions) > 1:
            st.error("neutral 不能和其他 emotion 混选。")
        else:
            save_manual_label(labels_path, tmp_path, key, selected_emotions, selected_intensities, reason)
            st.success(f"Saved manual label for {key}.")
            st.rerun()

    cot = result.get("cot") or ""
    if cot:
        st.divider()
        with st.expander(f"COT ({len(cot)} chars)", expanded=False):
            st.text_area("COT", value=cot, height=360, disabled=True)


if __name__ == "__main__":
    main()
