"""Preserve SDK callback identity without inventing another authentication path."""
import ast
import inspect
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter():
    value = SlackAdapter(PlatformConfig(enabled=True, token="synthetic-test-token"))
    value._app = MagicMock()
    value._app.client = AsyncMock()
    value._bot_user_id = "U_BOT"
    value._running = True
    value.handle_message = AsyncMock()
    value._resolve_user_name = AsyncMock(return_value="fixture")
    return value


def message() -> dict[str, Any]:
    return {"type": "message", "channel": "C_CHANNEL", "channel_type": "channel",
            "user": "U_USER", "text": "<@U_BOT> fixture", "ts": "1790000000.000001"}


@pytest.mark.asyncio
async def test_sdk_body_identity_survives_message_normalization(adapter):
    event = message()
    body = {"type": "event_callback", "team_id": "T_WORKSPACE", "api_app_id": "A_APP",
            "event_id": "Ev_EVENT", "event": event, "token": "synthetic-do-not-copy"}
    await adapter._handle_slack_message(event, socket_body=body)
    normalized = adapter.handle_message.call_args.args[0]
    assert normalized.metadata["slack_socket"] == {
        "type": "event_callback", "team_id": "T_WORKSPACE", "api_app_id": "A_APP", "event_id": "Ev_EVENT"}
    assert "synthetic-do-not-copy" not in repr(normalized.metadata)
    assert normalized.raw_message == event


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["handle_message_event", "handle_app_mention"])
async def test_registered_callback_forwards_bolt_body(name):
    # Execute the unchanged callback body from connect(); not a live Slack proof.
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(SlackAdapter.connect)))
    callback = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
    callback.decorator_list = []
    target = SimpleNamespace(_handle_slack_message=AsyncMock())
    namespace: dict[str, Any] = {"self": target}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), "<registered-slack-callback>", "exec"), namespace)
    event = message()
    body = {"event": event}
    await namespace[name](event=event, say=None, body=body)
    target._handle_slack_message.assert_awaited_once_with(event, socket_body=body)


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["different_event", "missing_app", "structured_id", "wrong_type"])
async def test_invalid_outer_context_is_not_carried(adapter, mutation):
    event = message()
    body = {"type": "event_callback", "team_id": "T_WORKSPACE", "api_app_id": "A_APP",
            "event_id": "Ev_EVENT", "event": dict(event)}
    if mutation == "different_event":
        body["event"]["user"] = "U_DIFFERENT"
    elif mutation == "missing_app":
        del body["api_app_id"]
    elif mutation == "structured_id":
        body["api_app_id"] = {"token": "synthetic-private"}
    else:
        body["type"] = "url_verification"
    await adapter._handle_slack_message(event, socket_body=body)
    assert adapter.handle_message.call_args.args[0].metadata == {}


@pytest.mark.asyncio
async def test_message_cannot_supply_its_own_socket_metadata(adapter):
    event = message()
    event["metadata"] = {"slack_socket": {"api_app_id": "A_FORGED"}}
    await adapter._handle_slack_message(event)
    assert adapter.handle_message.call_args.args[0].metadata == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["U_TRUSTED", "U_INTRUDER"])
async def test_adapter_context_reaches_only_post_authorization_hook(adapter, monkeypatch, user):
    from tests.gateway.test_post_gateway_authorization import runner_and_event
    from hermes_cli.plugins import get_plugin_manager

    runner, _ = runner_and_event(monkeypatch)
    event = message()
    event["user"] = user
    body = {"type": "event_callback", "team_id": "T_WORKSPACE", "api_app_id": "A_APP",
            "event_id": "Ev_EVENT", "event": event}
    reached = []

    async def consume(event):
        reached.append((event.source.user_id, event.metadata["slack_socket"]))
        return True

    monkeypatch.setattr(get_plugin_manager(), "_hooks", {"post_gateway_authorization": [consume]})
    adapter.handle_message = runner._handle_message
    await adapter._handle_slack_message(event, socket_body=body)
    if user == "U_TRUSTED":
        assert reached == [(user, {key: body[key] for key in ("type", "team_id", "api_app_id", "event_id")})]
    else:
        assert reached == []
    cast(AsyncMock, runner._handle_message_with_agent).assert_not_awaited()
