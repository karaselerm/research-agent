from typing import Any
from pathlib import Path
import base64
import io
import sys
import tarfile
import tempfile
import pytest
import httpx
import pandas as pd
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import FilePart, FileWithBytes, Message, Part, Role, TextPart

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import Agent


# A2A validation helpers - adapted from https://github.com/a2aproject/a2a-inspector/blob/main/backend/validators.py

def validate_agent_card(card_data: dict[str, Any]) -> list[str]:
    """Validate the structure and fields of an agent card."""
    errors: list[str] = []

    # Use a frozenset for efficient checking and to indicate immutability.
    required_fields = frozenset(
        [
            'name',
            'description',
            'url',
            'version',
            'capabilities',
            'defaultInputModes',
            'defaultOutputModes',
            'skills',
        ]
    )

    # Check for the presence of all required fields
    for field in required_fields:
        if field not in card_data:
            errors.append(f"Required field is missing: '{field}'.")

    # Check if 'url' is an absolute URL (basic check)
    if 'url' in card_data and not (
        card_data['url'].startswith('http://')
        or card_data['url'].startswith('https://')
    ):
        errors.append(
            "Field 'url' must be an absolute URL starting with http:// or https://."
        )

    # Check if capabilities is a dictionary
    if 'capabilities' in card_data and not isinstance(
        card_data['capabilities'], dict
    ):
        errors.append("Field 'capabilities' must be an object.")

    # Check if defaultInputModes and defaultOutputModes are arrays of strings
    for field in ['defaultInputModes', 'defaultOutputModes']:
        if field in card_data:
            if not isinstance(card_data[field], list):
                errors.append(f"Field '{field}' must be an array of strings.")
            elif not all(isinstance(item, str) for item in card_data[field]):
                errors.append(f"All items in '{field}' must be strings.")

    # Check skills array
    if 'skills' in card_data:
        if not isinstance(card_data['skills'], list):
            errors.append(
                "Field 'skills' must be an array of AgentSkill objects."
            )
        elif not card_data['skills']:
            errors.append(
                "Field 'skills' array is empty. Agent must have at least one skill if it performs actions."
            )

    return errors


def _validate_task(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'id' not in data:
        errors.append("Task object missing required field: 'id'.")
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append("Task object missing required field: 'status.state'.")
    return errors


def _validate_status_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'status' not in data or 'state' not in data.get('status', {}):
        errors.append(
            "StatusUpdate object missing required field: 'status.state'."
        )
    return errors


def _validate_artifact_update(data: dict[str, Any]) -> list[str]:
    errors = []
    if 'artifact' not in data:
        errors.append(
            "ArtifactUpdate object missing required field: 'artifact'."
        )
    elif (
        'parts' not in data.get('artifact', {})
        or not isinstance(data.get('artifact', {}).get('parts'), list)
        or not data.get('artifact', {}).get('parts')
    ):
        errors.append("Artifact object must have a non-empty 'parts' array.")
    return errors


def _validate_message(data: dict[str, Any]) -> list[str]:
    errors = []
    if (
        'parts' not in data
        or not isinstance(data.get('parts'), list)
        or not data.get('parts')
    ):
        errors.append("Message object must have a non-empty 'parts' array.")
    if 'role' not in data or data.get('role') != 'agent':
        errors.append("Message from agent must have 'role' set to 'agent'.")
    return errors


def validate_event(data: dict[str, Any]) -> list[str]:
    """Validate an incoming event from the agent based on its kind."""
    if 'kind' not in data:
        return ["Response from agent is missing required 'kind' field."]

    kind = data.get('kind')
    validators = {
        'task': _validate_task,
        'status-update': _validate_status_update,
        'artifact-update': _validate_artifact_update,
        'message': _validate_message,
    }

    validator = validators.get(str(kind))
    if validator:
        return validator(data)

    return [f"Unknown message kind received: '{kind}'."]


# A2A messaging helpers

async def send_text_message(text: str, url: str, context_id: str | None = None, streaming: bool = False):
    async with httpx.AsyncClient(timeout=10) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        events = [event async for event in client.send_message(msg)]

    return events


# A2A conformance tests

def test_agent_card(agent):
    """Validate agent card structure and required fields."""
    response = httpx.get(f"{agent}/.well-known/agent-card.json")
    assert response.status_code == 200, "Agent card endpoint must return 200"

    card_data = response.json()
    errors = validate_agent_card(card_data)

    assert not errors, f"Agent card validation failed:\n" + "\n".join(errors)

@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_message(agent, streaming):
    """Test that agent returns valid A2A message format."""
    events = await send_text_message("Hello", agent, streaming=streaming)

    all_errors = []
    for event in events:
        match event:
            case Message() as msg:
                errors = validate_event(msg.model_dump())
                all_errors.extend(errors)

            case (task, update):
                errors = validate_event(task.model_dump())
                all_errors.extend(errors)
                if update:
                    errors = validate_event(update.model_dump())
                    all_errors.extend(errors)

            case _:
                pytest.fail(f"Unexpected event type: {type(event)}")

    assert events, "Agent should respond with at least one event"
    assert not all_errors, f"Message validation failed:\n" + "\n".join(all_errors)

# Add your custom tests here


@pytest.mark.asyncio
async def test_submission_artifact_uses_expected_filename_and_final_chunk():
    class DummyUpdater:
        def __init__(self):
            self.calls = []

        async def add_artifact(self, parts, artifact_id=None, name=None, metadata=None, append=None, last_chunk=None, extensions=None):
            self.calls.append(
                {
                    "parts": parts,
                    "artifact_id": artifact_id,
                    "name": name,
                    "metadata": metadata,
                    "append": append,
                    "last_chunk": last_chunk,
                    "extensions": extensions,
                }
            )

    submission_df = pd.DataFrame(
        {
            "PassengerId": ["0013_01", "0018_01"],
            "Transported": ["True", "False"],
        }
    )
    updater = DummyUpdater()

    await Agent()._add_submission_artifact(updater, submission_df)

    assert len(updater.calls) == 1
    call = updater.calls[0]
    assert call["artifact_id"] == "submission"
    assert call["name"] == "submission.csv"
    assert call["append"] is False
    assert call["last_chunk"] is True
    assert len(call["parts"]) == 1

    part = call["parts"][0].root
    assert isinstance(part, FilePart)
    assert part.file.name == "submission.csv"
    assert part.file.mime_type == "text/csv"


def test_build_fallback_submission_is_valid_and_uses_test_ids():
    agent = Agent()
    sample_df = pd.DataFrame(
        {
            "PassengerId": ["0001_01", "0002_01", "0003_01"],
            "Transported": ["False", "False", "False"],
        }
    )
    test_df = pd.DataFrame(
        {
            "PassengerId": ["1010_01", "1011_01", "1012_01"],
            "Feature": [1, 2, 3],
        }
    )

    fallback_df = agent._build_fallback_submission(test_df=test_df, sample_df=sample_df)
    agent._validate_submission(fallback_df, sample_df)

    assert fallback_df.columns.tolist() == sample_df.columns.tolist()
    assert fallback_df["PassengerId"].tolist() == test_df["PassengerId"].tolist()
    assert set(fallback_df["Transported"].astype(str).str.lower().unique().tolist()).issubset({"true", "false"})


@pytest.mark.asyncio
async def test_run_submits_baseline_even_if_modeling_fails(monkeypatch):
    class DummyUpdater:
        def __init__(self):
            self.status_calls = []
            self.artifact_calls = []

        async def update_status(self, state, message):
            self.status_calls.append((state, message))

        async def add_artifact(self, parts, artifact_id=None, name=None, metadata=None, append=None, last_chunk=None, extensions=None):
            self.artifact_calls.append(
                {
                    "parts": parts,
                    "artifact_id": artifact_id,
                    "name": name,
                    "append": append,
                    "last_chunk": last_chunk,
                }
            )

    train_df = pd.DataFrame(
        {
            "PassengerId": ["0001_01", "0002_01", "0003_01"],
            "FeatureA": [1.0, 2.0, 3.0],
            "Transported": ["False", "True", "False"],
        }
    )
    test_df = pd.DataFrame(
        {
            "PassengerId": ["1001_01", "1002_01"],
            "FeatureA": [1.5, 2.5],
        }
    )
    sample_df = pd.DataFrame(
        {
            "PassengerId": ["xxxx_01", "yyyy_01"],
            "Transported": ["False", "False"],
        }
    )

    with tempfile.TemporaryDirectory(prefix="agent-test-") as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "home" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(data_dir / "train.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)
        sample_df.to_csv(data_dir / "sample_submission.csv", index=False)

        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
            tar.add(data_dir, arcname="home/data", recursive=True)
        tar_bytes = tar_buf.getvalue()

    msg = Message(
        kind="message",
        role=Role.user,
        parts=[
            Part(
                root=FilePart(
                    file=FileWithBytes(
                        bytes=base64.b64encode(tar_bytes).decode("ascii"),
                        name="competition.tar.gz",
                        mime_type="application/gzip",
                    )
                )
            )
        ],
        message_id=uuid4().hex,
    )

    agent = Agent()

    def _raise_modeling(*args, **kwargs):
        raise RuntimeError("forced modeling failure")

    monkeypatch.setattr(agent, "_solve_competition", _raise_modeling)

    updater = DummyUpdater()
    await agent.run(msg, updater)

    assert len(updater.artifact_calls) >= 1
    first_call = updater.artifact_calls[0]
    assert first_call["artifact_id"] == "submission"
    assert first_call["name"] == "submission.csv"
    assert first_call["append"] is False
    assert first_call["last_chunk"] is True
