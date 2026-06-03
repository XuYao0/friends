#!/usr/bin/env python3
"""Streamlit review UI for MELD-guided emotion labels.

The model output file is read-only in this app. Human corrections are stored in
an overlay JSON file so labeling jobs can continue writing their own results.
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


RESULTS_PATH = Path("workzone/meld_labeled_results.json")
MANUAL_LABELS_PATH = Path("workzone/meld_manual_labels.json")

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
MELD_AGREEMENTS = ["agree", "partial", "disagree"]


def read_json_file(path: Path, attempts: int = 5, delay: float = 0.08) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{path} must contain a JSON object")
            return data
        except FileNotFoundError:
            raise
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read {path}")


def load_results(path: Path) -> dict[str, Any]:
    try:
        data = read_json_file(path)
    except FileNotFoundError:
        st.error(f"Not found: {path}")
        st.stop()
    except (json.JSONDecodeError, ValueError) as exc:
        cached = st.session_state.get(f"last_results:{path}")
        if cached is not None:
            st.warning(f"{path} is being written right now; showing the last loaded snapshot.")
            return cached
        st.error(f"Could not parse {path}: {exc}")
        st.stop()

    if not isinstance(data.get("results"), list):
        st.error(f"{path} does not contain a results list.")
        st.stop()
    st.session_state[f"last_results:{path}"] = data
    return data


def load_manual_labels(path: Path = MANUAL_LABELS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {
            "metadata": {
                "source": str(RESULTS_PATH),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "labels": {},
        }
    try:
        data = read_json_file(path)
    except json.JSONDecodeError as exc:
        st.error(f"Could not parse {path}: {exc}")
        st.stop()
    labels = data.setdefault("labels", {})
    if not isinstance(labels, dict):
        st.error(f"{path} field labels must be a JSON object.")
        st.stop()
    data.setdefault("metadata", {})
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def result_key(result: dict[str, Any]) -> str:
    utterance = result.get("utterance")
    global_id = utterance.get("global_utterance_id") if isinstance(utterance, dict) else None
    if isinstance(global_id, int):
        return str(global_id)
    season = result.get("season")
    episode = result.get("episode")
    utterance_id = utterance.get("utterance_id") if isinstance(utterance, dict) else None
    return f"S{season}E{episode}U{utterance_id}"


def result_global_id(result: dict[str, Any]) -> int | None:
    utterance = result.get("utterance")
    if not isinstance(utterance, dict):
        return None
    global_id = utterance.get("global_utterance_id")
    return global_id if isinstance(global_id, int) else None


def result_label(result: dict[str, Any]) -> dict[str, Any]:
    label = result.get("label")
    return label if isinstance(label, dict) else {}


def result_meld(result: dict[str, Any]) -> dict[str, Any]:
    meld = result.get("meld")
    return meld if isinstance(meld, dict) else {}


def result_utterance(result: dict[str, Any]) -> dict[str, Any]:
    utterance = result.get("utterance")
    return utterance if isinstance(utterance, dict) else {}


def label_emotions(label: dict[str, Any]) -> list[str]:
    emotions = label.get("emotions")
    if isinstance(emotions, list):
        return [emotion for emotion in emotions if isinstance(emotion, str) and emotion in EMOTIONS]
    emotion = label.get("emotion")
    return [emotion] if isinstance(emotion, str) and emotion in EMOTIONS else []


def label_intensities(label: dict[str, Any], emotions: list[str]) -> list[str]:
    intensities = label.get("intensities")
    if isinstance(intensities, list) and len(intensities) == len(emotions):
        return [
            intensity if isinstance(intensity, str) and intensity in INTENSITIES else default_intensity(emotion)
            for emotion, intensity in zip(emotions, intensities)
        ]
    intensity = label.get("intensity")
    if isinstance(intensity, str) and intensity in INTENSITIES and emotions:
        return [intensity] + [default_intensity(emotion) for emotion in emotions[1:]]
    return [default_intensity(emotion) for emotion in emotions]


def default_intensity(emotion: str) -> str:
    return "none" if emotion == "neutral" else "medium"


def manual_label_for(manual_data: dict[str, Any], key: str) -> dict[str, Any]:
    labels = manual_data.get("labels")
    if not isinstance(labels, dict):
        return {}
    entry = labels.get(key)
    if not isinstance(entry, dict):
        return {}
    label = entry.get("manual_label")
    return label if isinstance(label, dict) else {}


def save_manual_label(
    key: str,
    result: dict[str, Any],
    emotions: list[str],
    intensities: list[str],
    reason: str,
    adjudication: str,
    notes: str,
) -> None:
    data = load_manual_labels()
    data["metadata"]["source"] = str(RESULTS_PATH)
    data["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["labels"][key] = {
        "season": result.get("season"),
        "episode": result.get("episode"),
        "global_utterance_id": result_global_id(result),
        "utterance_id": result_utterance(result).get("utterance_id"),
        "manual_label": {
            "emotions": emotions,
            "intensities": intensities,
            "emotion": emotions[0] if emotions else "neutral",
            "intensity": intensities[0] if intensities else "none",
            "reason": reason.strip(),
            "adjudication": adjudication,
            "notes": notes.strip(),
        },
        "model_label": result_label(result),
        "meld": result_meld(result),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(MANUAL_LABELS_PATH, data)


def delete_manual_label(key: str) -> None:
    data = load_manual_labels()
    labels = data.get("labels")
    if isinstance(labels, dict) and key in labels:
        del labels[key]
        data["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(MANUAL_LABELS_PATH, data)


def chip(text: str, color: str = "#1f2937") -> str:
    return (
        f'<span class="label-chip" style="background:{html.escape(color)}">'
        f"{html.escape(text)}</span>"
    )


def label_chips(emotions: Any, intensities: Any) -> str:
    chips: list[str] = []
    if isinstance(emotions, list) and isinstance(intensities, list):
        for emotion, intensity in zip(emotions, intensities):
            chips.append(chip(f"{emotion} / {intensity}"))
    return " ".join(chips) if chips else chip("(missing)")


def meld_chips(meld: dict[str, Any]) -> str:
    emotions = meld.get("emotion")
    sentiments = meld.get("sentiment")
    chips: list[str] = []
    if isinstance(emotions, list) and isinstance(sentiments, list):
        for emotion, sentiment in zip(emotions, sentiments):
            chips.append(chip(f"{emotion} / {sentiment}", "#0f766e"))
    elif isinstance(emotions, list):
        chips.extend(chip(str(emotion), "#0f766e") for emotion in emotions)
    return " ".join(chips) if chips else chip("(missing)", "#0f766e")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="stApp"] {
            font-size: 17px;
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            font-size: 1.02rem;
            line-height: 1.5;
        }
        textarea {
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }
        button {
            font-size: 1rem !important;
        }
        .info-block {
            border-left: 5px solid;
            border-radius: 6px;
            padding: 0.8rem 1rem;
            margin: 0.45rem 0 0.9rem 0;
        }
        .utterance-block {
            background: #f7fbff;
            border-color: #2563eb;
        }
        .model-block {
            background: #fff8eb;
            border-color: #d97706;
        }
        .manual-block {
            background: #f0fdf4;
            border-color: #16a34a;
        }
        .meld-block {
            background: #eefdf8;
            border-color: #0f766e;
        }
        .usage-block {
            background: #f8fafc;
            border-color: #64748b;
        }
        .block-title {
            color: #374151;
            font-size: 0.92rem;
            font-weight: 750;
            margin-bottom: 0.32rem;
            text-transform: uppercase;
        }
        .speaker {
            color: #1d4ed8;
            font-weight: 800;
        }
        .utterance-text {
            color: #111827;
            font-size: 1.2rem;
            line-height: 1.5;
        }
        .label-chip {
            display: inline-block;
            color: white;
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            margin: 0 0.25rem 0.25rem 0;
            font-size: 0.95rem;
            font-weight: 650;
        }
        .reason-text {
            color: #374151;
            font-size: 1rem;
            line-height: 1.55;
            margin-top: 0.35rem;
        }
        .meta-text {
            color: #475569;
            font-size: 0.94rem;
            line-height: 1.45;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_utterance(result: dict[str, Any]) -> None:
    utterance = result_utterance(result)
    speaker = html.escape(str(utterance.get("speaker", "")))
    text = html.escape(str(utterance.get("utterance", "")))
    inline_description = utterance.get("inline_description")
    inline_html = ""
    if isinstance(inline_description, list) and inline_description:
        inline_text = "; ".join(str(item) for item in inline_description)
        inline_html = f'<div class="reason-text">{html.escape(inline_text)}</div>'
    st.markdown(
        f"""
        <div class="info-block utterance-block">
          <div class="block-title">Utterance</div>
          <div class="utterance-text"><span class="speaker">{speaker}</span>: {text}</div>
          {inline_html}
          <div class="meta-text">
            S{result.get("season"):02d}E{result.get("episode"):02d};
            utterance_id={utterance.get("utterance_id")};
            global_id={utterance.get("global_utterance_id")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meld(result: dict[str, Any]) -> None:
    meld = result_meld(result)
    st.markdown(
        f"""
        <div class="info-block meld-block">
          <div class="block-title">MELD Weak Reference</div>
          <div>{meld_chips(meld)}</div>
          <div class="meta-text">
            status={html.escape(str(meld.get("match_status", "")))};
            score={html.escape(str(meld.get("match_score", "")))};
            text_score={html.escape(str(meld.get("match_text_score", "")))};
            group_size={html.escape(str(meld.get("meld_group_size", "")))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_label(result: dict[str, Any]) -> None:
    label = result_label(result)
    agreement = html.escape(str(label.get("meld_agreement", "")))
    reason = html.escape(str(label.get("reason", "")))
    st.markdown(
        f"""
        <div class="info-block model-block">
          <div class="block-title">Model Label</div>
          <div>{label_chips(label.get("emotions"), label.get("intensities"))}</div>
          <div class="meta-text">meld_agreement={agreement}</div>
          <div class="reason-text">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_manual_label(manual_label: dict[str, Any]) -> None:
    if not manual_label:
        return
    reason = html.escape(str(manual_label.get("reason", "")))
    adjudication = html.escape(str(manual_label.get("adjudication", "")))
    notes = html.escape(str(manual_label.get("notes", "")))
    notes_html = f'<div class="reason-text">{notes}</div>' if notes else ""
    st.markdown(
        f"""
        <div class="info-block manual-block">
          <div class="block-title">Manual Label</div>
          <div>{label_chips(manual_label.get("emotions"), manual_label.get("intensities"))}</div>
          <div class="meta-text">adjudication={adjudication}</div>
          <div class="reason-text">{reason}</div>
          {notes_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_usage(result: dict[str, Any]) -> None:
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return
    fields = [
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
    ]
    text = "; ".join(f"{field}={usage.get(field)}" for field in fields)
    st.markdown(
        f"""
        <div class="info-block usage-block">
          <div class="block-title">Usage</div>
          <div class="meta-text">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_matches_text(result: dict[str, Any], query: str) -> bool:
    if not query.strip():
        return True
    q = query.lower()
    utterance = result_utterance(result)
    label = result_label(result)
    fields = [
        utterance.get("speaker"),
        utterance.get("utterance"),
        label.get("reason"),
        result.get("season"),
        result.get("episode"),
        utterance.get("utterance_id"),
        utterance.get("global_utterance_id"),
    ]
    return any(q in str(field).lower() for field in fields if field is not None)


def filter_results(
    results: list[dict[str, Any]],
    agreement_filter: list[str],
    status_filter: list[str],
    group_filter: str,
    manual_filter: str,
    manual_data: dict[str, Any],
    query: str,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for result in results:
        label = result_label(result)
        meld = result_meld(result)
        if agreement_filter and label.get("meld_agreement") not in agreement_filter:
            continue
        if status_filter and meld.get("match_status") not in status_filter:
            continue
        group_size = meld.get("meld_group_size")
        if group_filter == "single" and group_size != 1:
            continue
        if group_filter == "multi" and not (isinstance(group_size, int) and group_size > 1):
            continue
        has_manual = bool(manual_label_for(manual_data, result_key(result)))
        if manual_filter == "manual only" and not has_manual:
            continue
        if manual_filter == "unreviewed only" and has_manual:
            continue
        if not result_matches_text(result, query):
            continue
        filtered.append(result)
    return filtered


def set_current_index(index: int, total: int) -> None:
    if total <= 0:
        st.session_state["current_result_index"] = 0
        return
    st.session_state["current_result_index"] = min(max(index, 0), total - 1)


def main() -> None:
    st.set_page_config(page_title="MELD Emotion Label Review", layout="wide")
    inject_css()
    st.title("MELD Emotion Label Review")

    data = load_results(RESULTS_PATH)
    manual_data = load_manual_labels()
    raw_results = [result for result in data["results"] if isinstance(result, dict)]
    if not raw_results:
        st.info("No model results loaded yet.")
        st.stop()

    with st.sidebar:
        st.header("Filters")
        agreement_filter = st.multiselect("MELD agreement", MELD_AGREEMENTS, default=[])
        match_statuses = sorted(
            {
                str(result_meld(result).get("match_status"))
                for result in raw_results
                if result_meld(result).get("match_status") is not None
            }
        )
        status_filter = st.multiselect("Match status", match_statuses, default=[])
        group_filter = st.radio("MELD group", ["all", "single", "multi"], horizontal=True)
        manual_filter = st.radio("Review state", ["all", "manual only", "unreviewed only"])
        query = st.text_input("Search speaker/text/reason/id", value="")

    filtered_results = filter_results(
        raw_results,
        agreement_filter,
        status_filter,
        group_filter,
        manual_filter,
        manual_data,
        query,
    )
    if not filtered_results:
        st.warning("No results match the current filters.")
        st.stop()

    if "current_result_index" not in st.session_state:
        st.session_state["current_result_index"] = 0
    set_current_index(st.session_state["current_result_index"], len(filtered_results))
    current_index = st.session_state["current_result_index"]
    result = filtered_results[current_index]
    key = result_key(result)

    prev_col, next_col, pos_col, jump_col = st.columns([1, 1, 1.4, 3])
    with prev_col:
        if st.button("Previous", disabled=current_index == 0, use_container_width=True):
            set_current_index(current_index - 1, len(filtered_results))
            st.rerun()
    with next_col:
        if st.button("Next", disabled=current_index == len(filtered_results) - 1, use_container_width=True):
            set_current_index(current_index + 1, len(filtered_results))
            st.rerun()
    with pos_col:
        st.metric("Result", f"{current_index + 1} / {len(filtered_results)}")
    with jump_col:
        jump_number = st.number_input(
            "Jump to filtered result number",
            min_value=1,
            max_value=len(filtered_results),
            value=current_index + 1,
            step=1,
        )
        if st.button("Jump", use_container_width=True):
            set_current_index(int(jump_number) - 1, len(filtered_results))
            st.rerun()

    manual_labels = manual_data.get("labels") if isinstance(manual_data.get("labels"), dict) else {}
    st.caption(
        f"Loaded {len(raw_results)} model results from {RESULTS_PATH}; "
        f"showing {len(filtered_results)} after filters; "
        f"manual labels: {len(manual_labels)} saved to {MANUAL_LABELS_PATH}."
    )

    render_utterance(result)

    left_col, right_col = st.columns(2)
    with left_col:
        render_meld(result)
    with right_col:
        render_model_label(result)

    manual_label = manual_label_for(manual_data, key)
    render_manual_label(manual_label)
    render_usage(result)

    st.divider()
    st.subheader("Manual Annotation")
    model_label = result_label(result)
    default_emotions = label_emotions(manual_label) or label_emotions(model_label) or ["neutral"]
    default_intensities = label_intensities(manual_label, default_emotions)
    if not manual_label:
        default_intensities = label_intensities(model_label, default_emotions)

    selected_emotions = st.multiselect(
        "Emotions",
        EMOTIONS,
        default=default_emotions,
        key=f"emotions_{key}",
    )
    if "neutral" in selected_emotions and len(selected_emotions) > 1:
        st.warning("neutral cannot be mixed with other emotions.")

    selected_intensities: list[str] = []
    if selected_emotions:
        st.write("Intensity")
        intensity_cols = st.columns(min(3, len(selected_emotions)))
        default_by_emotion = dict(zip(default_emotions, default_intensities))
        for index, emotion in enumerate(selected_emotions):
            options = ["none"] if emotion == "neutral" else ["low", "medium", "high"]
            default_value = default_by_emotion.get(emotion, default_intensity(emotion))
            if default_value not in options:
                default_value = default_intensity(emotion)
            with intensity_cols[index % len(intensity_cols)]:
                selected_intensities.append(
                    st.radio(
                        emotion,
                        options,
                        index=options.index(default_value),
                        horizontal=True,
                        key=f"intensity_{key}_{emotion}",
                    )
                )

    reason = st.text_area(
        "Reason",
        value=str(manual_label.get("reason") or model_label.get("reason") or ""),
        height=110,
        key=f"reason_{key}",
    )
    adjudication = st.radio(
        "Adjudication",
        ["accept model", "revise model", "uncertain", "exclude"],
        index=["accept model", "revise model", "uncertain", "exclude"].index(
            str(manual_label.get("adjudication", "accept model"))
            if manual_label.get("adjudication") in {"accept model", "revise model", "uncertain", "exclude"}
            else "accept model"
        ),
        horizontal=True,
        key=f"adjudication_{key}",
    )
    notes = st.text_area(
        "Notes",
        value=str(manual_label.get("notes", "")),
        height=80,
        key=f"notes_{key}",
    )

    save_col, delete_col = st.columns([1, 1])
    with save_col:
        if st.button("Save Manual Label", type="primary", use_container_width=True):
            if not selected_emotions:
                st.error("Select at least one emotion.")
            elif "neutral" in selected_emotions and len(selected_emotions) > 1:
                st.error("neutral cannot be mixed with other emotions.")
            elif len(selected_intensities) != len(selected_emotions):
                st.error("Each emotion must have one intensity.")
            elif not reason.strip():
                st.error("Reason is required.")
            else:
                save_manual_label(
                    key,
                    result,
                    selected_emotions,
                    selected_intensities,
                    reason,
                    adjudication,
                    notes,
                )
                st.success(f"Saved manual label for key {key}.")
                st.rerun()
    with delete_col:
        if st.button("Delete Manual Label", disabled=not manual_label, use_container_width=True):
            delete_manual_label(key)
            st.success(f"Deleted manual label for key {key}.")
            st.rerun()

    cot = result.get("cot")
    if isinstance(cot, str) and cot.strip():
        with st.expander(f"COT ({len(cot)} chars)", expanded=False):
            st.text_area("COT", value=cot, height=360, disabled=True, key=f"cot_{key}")

    with st.expander("Raw result", expanded=False):
        st.json(result)


if __name__ == "__main__":
    main()
