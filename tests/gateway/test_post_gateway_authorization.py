"""Post-authorization gateway seam used by authenticated lifecycle consumers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True)},
    )
    runner.adapters = {Platform.SLACK: SimpleNamespace(send=AsyncMock())}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner


def _event(user_id="U_ALLOWED"):
    return MessageEvent(
        text="implement: bounded local candidate",
        message_id="2000000000.000001",
        source=SessionSource(
            platform=Platform.SLACK,
            user_id=user_id,
            chat_id="C0BSXSM76GG",
            user_name="Authorized Human",
            chat_type="group",
        ),
        raw_message={
            "type": "message",
            "user": user_id,
            "channel": "C0BSXSM76GG",
            "team": "T_TEST",
            "ts": "2000000000.000001",
            "text": "implement: bounded local candidate",
        },
        metadata={
            "slack_socket": {
                "type": "event_callback",
                "team_id": "T_TEST",
                "api_app_id": "A_TEST",
                "event_id": "Ev_T1_1",
            }
        },
    )


@pytest.mark.asyncio
async def test_post_gateway_authorization_awaits_consumer_and_short_circuits(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_ALLOWED")
    seen = []
    expected = object()

    async def consume(**kwargs):
        seen.append(kwargs["event"])
        return {"action": "handled", "result": expected}

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: [consume(**kwargs)] if name == "post_gateway_authorization" else [],
    )

    result = await _runner()._handle_message(_event())

    assert result is expected
    assert len(seen) == 1
    assert seen[0].metadata["slack_socket"]["event_id"] == "Ev_T1_1"


@pytest.mark.asyncio
async def test_post_gateway_authorization_sync_failure_fails_closed(monkeypatch):
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_ALLOWED")
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="failing-consumer"), manager)

    def consume(**_kwargs):
        raise RuntimeError("synchronous consumer failure")

    context.register_hook("post_gateway_authorization", consume)
    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    assert await _runner()._handle_message(_event()) is None


@pytest.mark.asyncio
async def test_post_gateway_authorization_never_runs_for_unauthorized_sender(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_ALLOWED")
    called = False

    async def consume(**_kwargs):
        nonlocal called
        called = True
        return {"action": "handled"}

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: [consume(**kwargs)] if name == "post_gateway_authorization" else [],
    )

    assert await _runner()._handle_message(_event("U_INTRUDER")) is None
    assert called is False


@pytest.mark.asyncio
async def test_slack_adapter_preserves_server_socket_context():
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter(PlatformConfig(enabled=True, extra={"require_mention": False}))
    adapter._bot_user_id = "U_BOT"
    adapter._app = MagicMock()
    adapter._resolve_user_name = AsyncMock(return_value="Authorized Human")
    adapter._get_channel_name = AsyncMock(return_value="agent-mercy")
    adapter.handle_message = AsyncMock()
    socket_context = {
        "type": "event_callback",
        "team_id": "T_TEST",
        "api_app_id": "A_TEST",
        "event_id": "Ev_T1_1",
    }

    await adapter._handle_slack_message(_event().raw_message, socket_context=socket_context)

    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.metadata["slack_socket"] == socket_context
