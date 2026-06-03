#!/usr/bin/env python3
"""Streamlit review UI for tmp.json emotion labels.

This app never writes to workzone/tmp.json. Manual corrections are stored in a
separate overlay file so the labeling script can keep updating tmp.json safely.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


TMP_PATHS = [Path("workzone/tmp.json"), Path("workzone/tmp_1.json")]
XY_LABELS_PATH = Path("workzone/xy_labels.json")
MELD_RECORDS_PATH = Path("workzone/friends_records_with_meld.jsonl")

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


def load_xy_labels() -> dict[str, Any]:
    if not XY_LABELS_PATH.exists():
        return {
            "metadata": {
                "source": [str(path) for path in TMP_PATHS],
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "labels": {},
        }
    try:
        data = read_json_file(XY_LABELS_PATH)
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse {XY_LABELS_PATH}: {exc}")
        st.stop()
    if not isinstance(data, dict):
        st.error(f"{XY_LABELS_PATH} must contain a JSON object.")
        st.stop()
    labels = data.setdefault("labels", {})
    if not isinstance(labels, dict):
        st.error(f"{XY_LABELS_PATH} field labels must be a JSON object.")
        st.stop()
    data.setdefault("metadata", {})
    return data


def load_meld_labels(path: Path = MELD_RECORDS_PATH) -> dict[int, dict[str, Any]]:
    labels: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                st.error(f"Could not parse {path}:{line_no}: {exc}")
                st.stop()
            meld = record.get("meld")
            content = record.get("content")
            if not isinstance(meld, dict) or not isinstance(content, dict):
                continue
            global_id = content.get("global_utterance_id")
            if isinstance(global_id, int):
                labels[global_id] = meld
    return labels


def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="stApp"] {
            font-size: 18px;
        }
        .stMarkdown, .stText, .stCaption, .stRadio, .stMultiSelect, .stTextArea, .stNumberInput {
            font-size: 1.05rem;
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            font-size: 1.05rem;
            line-height: 1.55;
        }
        label,
        div[role="radiogroup"] label,
        div[data-baseweb="select"] {
            font-size: 1.02rem;
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
        .model-block-2 {
            background: #f8f5ff;
            border-color: #7c3aed;
        }
        .xy-block {
            background: #f0fdf4;
            border-color: #16a34a;
        }
        .meld-block {
            background: #eefdf8;
            border-color: #0f766e;
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


def save_xy_label(utterance_id: int, emotions: list[str], intensities: list[str], reason: str) -> None:
    data = load_xy_labels()
    data["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    first_emotion = emotions[0] if emotions else "neutral"
    first_intensity = intensities[0] if intensities else "none"
    data["labels"][str(utterance_id)] = {
        "xy_label": {
            "emotions": emotions,
            "intensities": intensities,
            "emotion": first_emotion,
            "intensity": first_intensity,
            "reason": reason.strip(),
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    XY_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = XY_LABELS_PATH.with_suffix(XY_LABELS_PATH.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(XY_LABELS_PATH)


def result_utterance_id(result: dict[str, Any]) -> int | None:
    utterance = result.get("utterance")
    if not isinstance(utterance, dict):
        return None
    utterance_id = utterance.get("utterance_id")
    return utterance_id if isinstance(utterance_id, int) else None


def result_global_utterance_id(result: dict[str, Any]) -> int | None:
    utterance = result.get("utterance")
    if not isinstance(utterance, dict):
        return None
    global_id = utterance.get("global_utterance_id")
    if isinstance(global_id, int):
        return global_id
    return result_utterance_id(result)


def result_by_utterance_id(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        utterance_id = result_utterance_id(result)
        if utterance_id is not None:
            indexed[utterance_id] = result
    return indexed


def format_model_label(label: dict[str, Any]) -> str:
    emotions = label.get("emotions", [])
    intensities = label.get("intensities", [])
    pairs = []
    if isinstance(emotions, list) and isinstance(intensities, list):
        for emotion, intensity in zip(emotions, intensities):
            pairs.append(f"{emotion} / {intensity}")
    return ", ".join(pairs) if pairs else "(missing)"


def label_pairs_html(emotions: Any, intensities: Any) -> str:
    chips = []
    if isinstance(emotions, list) and isinstance(intensities, list):
        for emotion, intensity in zip(emotions, intensities):
            chips.append(f'<span class="label-chip">{emotion} / {intensity}</span>')
    return " ".join(chips) if chips else '<span class="label-chip">(missing)</span>'


def meld_pairs_html(emotions: Any, sentiments: Any) -> str:
    chips = []
    if isinstance(emotions, list) and isinstance(sentiments, list):
        for emotion, sentiment in zip(emotions, sentiments):
            chips.append(f'<span class="label-chip">{emotion} / {sentiment}</span>')
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
        cleaned = [intensity if intensity in INTENSITIES else "medium" for intensity in intensities]
        return cleaned
    intensity = xy_label.get("intensity")
    if intensity in INTENSITIES and emotions:
        return [intensity] + ["medium"] * (len(emotions) - 1)
    return ["none" if emotion == "neutral" else "medium" for emotion in emotions]


def set_current_id(utterance_id: int, valid_ids: list[int]) -> None:
    if not valid_ids:
        return
    min_id = min(valid_ids)
    max_id = max(valid_ids)
    st.session_state["current_utterance_id"] = min(max(utterance_id, min_id), max_id)


def first_available_result(current_results: list[tuple[Path, dict[str, Any] | None]]) -> dict[str, Any]:
    for _, result in current_results:
        if result is not None:
            return result
    return {}


def first_result_for_id(
    indexed_by_path: list[tuple[Path, dict[int, dict[str, Any]]]],
    utterance_id: int,
) -> dict[str, Any]:
    for _, indexed in indexed_by_path:
        result = indexed.get(utterance_id)
        if result is not None:
            return result
    return {}


def ids_with_meld_labels(
    valid_ids: list[int],
    indexed_by_path: list[tuple[Path, dict[int, dict[str, Any]]]],
    meld_labels: dict[int, dict[str, Any]],
) -> list[int]:
    ids: list[int] = []
    for utterance_id in valid_ids:
        result = first_result_for_id(indexed_by_path, utterance_id)
        global_id = result_global_utterance_id(result)
        if global_id is not None and global_id in meld_labels:
            ids.append(utterance_id)
    return ids


def previous_next_meld_ids(
    valid_ids: list[int],
    meld_ids: list[int],
    current_id: int,
) -> tuple[int | None, int | None]:
    if not meld_ids:
        return None, None
    current_pos = valid_ids.index(current_id)
    previous_id = next(
        (utterance_id for utterance_id in reversed(meld_ids) if valid_ids.index(utterance_id) < current_pos),
        None,
    )
    next_id = next(
        (utterance_id for utterance_id in meld_ids if valid_ids.index(utterance_id) > current_pos),
        None,
    )
    return previous_id, next_id


def render_model_result(path: Path, result: dict[str, Any] | None, block_class: str) -> None:
    if result is None:
        st.markdown(
            f"""
            <div class="info-block {block_class}">
              <div class="block-title">{path.name}</div>
              <div class="reason-text">No label for this utterance id.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    label = result.get("label", {})
    model_label = label if isinstance(label, dict) else {}
    st.markdown(
        f"""
        <div class="info-block {block_class}">
          <div class="block-title">{path.name}</div>
          <div>{label_pairs_html(model_label.get("emotions"), model_label.get("intensities"))}</div>
          <div class="reason-text">{model_label.get("reason", "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meld_label(meld_label: dict[str, Any] | None, global_id: int | None) -> None:
    if meld_label is None:
        st.markdown(
            f"""
            <div class="info-block meld-block">
              <div class="block-title">MELD Label</div>
              <div class="reason-text">No MELD match for global utterance id {global_id}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    status = meld_label.get("match_status", "")
    score = meld_label.get("match_score", "")
    text_score = meld_label.get("match_text_score", "")
    group_size = meld_label.get("meld_group_size", "")
    st.markdown(
        f"""
        <div class="info-block meld-block">
          <div class="block-title">MELD Label</div>
          <div>{meld_pairs_html(meld_label.get("emotion"), meld_label.get("sentiment"))}</div>
          <div class="reason-text">
            status={status}; score={score}; text_score={text_score}; group_size={group_size}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_available_tmp_datasets() -> list[tuple[Path, dict[str, Any]]]:
    existing_paths = [path for path in TMP_PATHS if path.exists()]
    if not existing_paths:
        st.error(f"No tmp files found: {', '.join(str(path) for path in TMP_PATHS)}")
        st.stop()
    return [(path, load_tmp(path)) for path in existing_paths]


def main() -> None:
    st.set_page_config(page_title="Friends Emotion Review", layout="wide")
    inject_css()
    st.title("Friends S01E01 Emotion Review")

    tmp_datasets = load_available_tmp_datasets()
    xy_data = load_xy_labels()
    meld_labels = load_meld_labels()
    indexed_by_path: list[tuple[Path, dict[int, dict[str, Any]]]] = []
    for path, tmp_data in tmp_datasets:
        results = [item for item in tmp_data["results"] if isinstance(item, dict)]
        indexed_by_path.append((path, result_by_utterance_id(results)))
    valid_ids = sorted(set().union(*(set(indexed) for _, indexed in indexed_by_path)))
    if not valid_ids:
        st.info("No results loaded yet.")
        st.stop()

    if "current_utterance_id" not in st.session_state:
        st.session_state["current_utterance_id"] = valid_ids[0]
    if st.session_state["current_utterance_id"] not in valid_ids:
        st.session_state["current_utterance_id"] = valid_ids[-1]

    current_id = st.session_state["current_utterance_id"]
    current_pos = valid_ids.index(current_id)
    meld_valid_ids = ids_with_meld_labels(valid_ids, indexed_by_path, meld_labels)
    previous_meld_id, next_meld_id = previous_next_meld_ids(valid_ids, meld_valid_ids, current_id)

    prev_col, next_col, current_col, prev_meld_col, next_meld_col, jump = st.columns([1, 1, 1, 1, 1, 3])
    with prev_col:
        if st.button("上一个", disabled=current_pos == 0, use_container_width=True):
            set_current_id(valid_ids[current_pos - 1], valid_ids)
            st.rerun()
    with next_col:
        if st.button("下一个", disabled=current_pos == len(valid_ids) - 1, use_container_width=True):
            set_current_id(valid_ids[current_pos + 1], valid_ids)
            st.rerun()
    with current_col:
        st.metric("当前 ID", current_id)
    with prev_meld_col:
        if st.button("上一个 MELD", disabled=previous_meld_id is None, use_container_width=True):
            if previous_meld_id is not None:
                set_current_id(previous_meld_id, valid_ids)
            st.rerun()
    with next_meld_col:
        if st.button("下一个 MELD", disabled=next_meld_id is None, use_container_width=True):
            if next_meld_id is not None:
                set_current_id(next_meld_id, valid_ids)
            st.rerun()
    with jump:
        target_id = st.number_input(
            "指定 id 跳转",
            min_value=min(valid_ids),
            max_value=max(valid_ids),
            value=current_id,
            step=1,
        )
        if st.button("跳转", use_container_width=True):
            set_current_id(int(target_id), valid_ids)
            st.rerun()

    counts = ", ".join(f"{path.name}: {len(indexed)}" for path, indexed in indexed_by_path)
    st.caption(
        f"Loaded labels from {counts}; tmp ids with MELD: {len(meld_valid_ids)}; "
        f"MELD labels: {len(meld_labels)}; "
        f"manual labels are saved to {XY_LABELS_PATH}."
    )

    current_results = [(path, indexed.get(current_id)) for path, indexed in indexed_by_path]
    result = first_available_result(current_results)
    utterance = result.get("utterance", {})
    label = result.get("label", {})
    global_id = result_global_utterance_id(result)
    meld_label = meld_labels.get(global_id) if global_id is not None else None
    xy_entry = xy_data["labels"].get(str(current_id), {})
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
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_cols = st.columns(2)
    for index, (path, current_result) in enumerate(current_results):
        with model_cols[index % 2]:
            render_model_result(path, current_result, "model-block" if index == 0 else "model-block-2")

    render_meld_label(meld_label, global_id)

    if xy_label:
        current_xy_emotions = xy_emotions(xy_label)
        current_xy_intensities = xy_intensities(xy_label, current_xy_emotions)
        st.markdown(
            f"""
            <div class="info-block xy-block">
              <div class="block-title">XY Label</div>
              <div>{label_pairs_html(current_xy_emotions, current_xy_intensities)}</div>
              <div class="reason-text">{xy_label.get("reason", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Manual Annotation")
    default_emotions = xy_emotions(xy_label)
    if not default_emotions and isinstance(label.get("emotions"), list):
        default_emotions = [emotion for emotion in label["emotions"] if emotion in EMOTIONS]
    if not default_emotions:
        default_emotions = ["neutral"]
    default_intensities = xy_intensities(xy_label, default_emotions)
    selected_emotions = st.multiselect(
        "Emotions",
        EMOTIONS,
        default=default_emotions,
        key=f"emotions_{current_id}",
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
                        key=f"intensity_{current_id}_{emotion}",
                    )
                )
    reason = st.text_area(
        "Reason",
        value=xy_label.get("reason") or (label.get("reason", "") if isinstance(label, dict) else ""),
        height=100,
        key=f"reason_{current_id}",
    )

    if st.button("提交标注", type="primary"):
        if not reason.strip():
            st.error("Reason 不能为空。")
        elif not selected_emotions:
            st.error("至少选择一个 emotion。")
        elif "neutral" in selected_emotions and len(selected_emotions) > 1:
            st.error("neutral 不能和其他 emotion 混选。")
        else:
            save_xy_label(current_id, selected_emotions, selected_intensities, reason)
            st.success(f"Saved xy_label for utterance {current_id}.")
            st.rerun()

    cot_results = [(path, current_result) for path, current_result in current_results if current_result and current_result.get("cot")]
    if cot_results:
        st.divider()
        for path, current_result in cot_results:
            cot = current_result.get("cot") or ""
            with st.expander(f"{path.name} COT ({len(cot)} chars)", expanded=False):
                st.text_area(
                    f"{path.name} COT",
                    value=cot,
                    height=360,
                    disabled=True,
                    key=f"cot_{path.name}_{current_id}",
                )


if __name__ == "__main__":
    main()
