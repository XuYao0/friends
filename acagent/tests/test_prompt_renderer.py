from acagent.llm import PromptRenderer
from acagent.schemas import MemoryItem, Utterance


def test_prompt_renderer_keeps_strings_as_plain_text() -> None:
    renderer = PromptRenderer()

    rendered = renderer.render_text("Reason: $reason", {"reason": "plain text"})

    assert rendered == "Reason: plain text"


def test_prompt_renderer_formats_structured_values_as_stable_json() -> None:
    renderer = PromptRenderer()

    rendered = renderer.render_text(
        "Payload:\n$payload",
        {
            "payload": {
                "speaker": "Monica",
                "items": [MemoryItem(text="Worries about being judged.")],
            }
        },
    )

    assert '"items": [' in rendered
    assert '"speaker": "Monica"' in rendered
    assert '"text": "Worries about being judged."' in rendered


def test_prompt_renderer_formats_dataclasses_as_json() -> None:
    renderer = PromptRenderer()
    utterance = Utterance(
        episode_id="S01E01",
        scene_id="S01E01_SC001",
        utterance_id="S01E01_U000001",
        turn_index=1,
        speaker="Monica",
        text="There's nothing to tell!",
        stage_direction="defensive",
    )

    rendered = renderer.render_text("Current:\n$current", {"current": utterance})

    assert '"speaker": "Monica"' in rendered
    assert '"stage_direction": "defensive"' in rendered
    assert '"utterance_id": "S01E01_U000001"' in rendered


def test_prompt_renderer_renders_files(tmp_path) -> None:
    path = tmp_path / "prompt.md"
    path.write_text("Speaker: $speaker\nContext: $context", encoding="utf-8")

    rendered = PromptRenderer().render_file(
        path,
        {
            "speaker": "Rachel",
            "context": {"scene_id": "S01E01_SC001"},
        },
    )

    assert "Speaker: Rachel" in rendered
    assert '"scene_id": "S01E01_SC001"' in rendered
