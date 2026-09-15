"""Exercise registered routes behind the existing, unmocked sender authorization."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.plugins import get_plugin_manager


def runner_and_event(monkeypatch, user="U_TRUSTED"):
    from gateway.run import GatewayRunner

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_TRUSTED")
    for name in ("GATEWAY_ALLOW_ALL_USERS", "SLACK_ALLOW_ALL_USERS", "SLACK_ALLOW_BOTS"):
        monkeypatch.delenv(name, raising=False)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True)})
    runner.adapters = {Platform.SLACK: SimpleNamespace(send=AsyncMock())}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    runner._handle_message_with_agent = AsyncMock(return_value="ordinary chat")
    runner._enqueue_fifo = MagicMock()
    event = MessageEvent(text="fixture task", message_id="1790000000.000001",
        source=SessionSource(platform=Platform.SLACK, user_id=user,
                             chat_id="C0BSXSM76GG", chat_type="group"))
    return runner, event


async def dispatch(runner, event, busy):
    if busy:
        assert await runner._handle_active_session_busy_message(event, "fixture-session") is True
    else:
        await runner._handle_message(event)
    runner._handle_message_with_agent.assert_not_awaited()
    runner._enqueue_fifo.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["cold", "busy"])
@pytest.mark.parametrize("user", ["U_TRUSTED", "U_INTRUDER", None])
async def test_only_authorized_sender_reaches_registered_route(monkeypatch, busy, user):
    runner, event = runner_and_event(monkeypatch, user)
    reached = []

    async def consume(event):
        reached.append(event.source.user_id)
        return True

    monkeypatch.setattr(get_plugin_manager(), "_hooks", {"post_gateway_authorization": [consume]})
    await dispatch(runner, event, busy)
    assert reached == ([user] if user == "U_TRUSTED" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True], ids=["cold", "busy"])
@pytest.mark.parametrize("failure", ["exception", "invalid_result", "timeout"])
async def test_route_failure_never_falls_back_to_chat(monkeypatch, caplog, busy, failure):
    runner, event = runner_and_event(monkeypatch)
    reached, cleaned = [], []
    real_wait_for = asyncio.wait_for

    async def short_wait_for(awaitable, timeout):
        return await real_wait_for(awaitable, timeout=0.02)

    async def consume(event):
        reached.append(event.message_id)
        try:
            if failure == "exception":
                raise ValueError("synthetic-private-diagnostic")
            if failure == "invalid_result":
                return {"handled": True}
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    monkeypatch.setattr(asyncio, "wait_for", short_wait_for)
    monkeypatch.setattr(get_plugin_manager(), "_hooks", {"post_gateway_authorization": [consume]})
    await dispatch(runner, event, busy)
    assert reached == [event.message_id]
    assert cleaned == [True]
    assert "failed closed" in caplog.text
    assert "synthetic-private-diagnostic" not in caplog.text


@pytest.mark.asyncio
async def test_internal_message_does_not_enter_human_route(monkeypatch):
    runner, event = runner_and_event(monkeypatch)
    event.internal = True
    callback = AsyncMock(return_value=True)
    monkeypatch.setattr(get_plugin_manager(), "_hooks", {"post_gateway_authorization": [callback]})
    assert await runner._consume_authorized_gateway_message(event) is False
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_consumer_prevents_duplicate_handoff(monkeypatch):
    _, event = runner_and_event(monkeypatch)
    callbacks = [AsyncMock(return_value=None), AsyncMock(return_value=True), AsyncMock(return_value=True)]
    monkeypatch.setattr(get_plugin_manager(), "_hooks", {"post_gateway_authorization": callbacks})
    assert await get_plugin_manager().consume_gateway_message(event) is True
    callbacks[0].assert_awaited_once_with(event=event)
    callbacks[1].assert_awaited_once_with(event=event)
    callbacks[2].assert_not_awaited()


@pytest.mark.asyncio
async def test_unmatched_route_preserves_existing_dispatch(monkeypatch):
    _, event = runner_and_event(monkeypatch)
    callbacks = [AsyncMock(return_value=None), AsyncMock(return_value=False)]
    monkeypatch.setattr(get_plugin_manager(), "_hooks", {"post_gateway_authorization": callbacks})
    assert await get_plugin_manager().consume_gateway_message(event) is False
    for callback in callbacks:
        callback.assert_awaited_once_with(event=event)
