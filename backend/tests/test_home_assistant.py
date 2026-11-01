import pytest
from app.services.home_assistant_service import home_assistant_service
from app.services.memory_readiness_service import memory_readiness_service


def test_home_assistant_unknown_query():
    resp = home_assistant_service.process_command("test-user-id", "Hello there!")
    assert resp.intent == "UNKNOWN"
    assert "help manage your memories" in resp.message
    assert len(resp.suggested_chips) > 0


def test_home_assistant_show_memories_no_memories():
    resp = home_assistant_service.process_command("non-existent-user", "Show my memories")
    assert resp.intent == "LIST_MEMORIES"
    assert "haven't created any memories yet" in resp.message or "memory" in resp.message


def test_home_assistant_show_vaults_no_vaults():
    resp = home_assistant_service.process_command("non-existent-user", "Show my vaults")
    assert resp.intent == "LIST_VAULTS"
    assert "don't have any rooms" in resp.message or "vault" in resp.message


def test_memory_readiness_non_existent():
    from unittest.mock import patch, MagicMock
    mock_res = MagicMock(data=[])
    with patch("app.services.memory_readiness_service.get_supabase_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = mock_res
        mock_get_client.return_value = mock_client
        with pytest.raises(ValueError):
            memory_readiness_service.evaluate_memory_readiness("test-user", "00000000-0000-0000-0000-000000000000")


def test_home_assistant_contextual_suggestions():
    from unittest.mock import patch
    from app.services.home_assistant_service import HomeAssistantService
    fake_mem = {
        "id": "mem-123",
        "title": "Chennai Trip",
        "description": "Visited Marina beach and temples.",
        "memory_date": "2026-10-28",
        "media": [{"id": "m1", "media_type": "image"}],
    }
    with patch.object(HomeAssistantService, "_get_memory_by_id", return_value=fake_mem):
        resp = home_assistant_service.process_command(
            user_id="00000000-0000-0000-0000-000000000001",
            query="What should I add?",
            context_memory_id="mem-123",
        )
        assert resp.intent == "SUGGEST_MEMORY_MEDIA"
        assert "Chennai Trip" in resp.message
        assert len(resp.actions) == 1
        assert resp.actions[0].type == "ADD_MEDIA"


def test_home_assistant_contextual_summarize():
    from unittest.mock import patch
    from app.services.home_assistant_service import HomeAssistantService
    fake_mem = {
        "id": "mem-123",
        "title": "Chennai",
        "description": "Created via AI Assistant on October 28, 2026.",
        "memory_date": "2026-10-28",
        "media": [],
    }
    with patch.object(HomeAssistantService, "_get_memory_by_id", return_value=fake_mem):
        resp = home_assistant_service.process_command(
            user_id="00000000-0000-0000-0000-000000000001",
            query="Summarize this memory",
            context_memory_id="mem-123",
        )
        assert resp.intent == "SUMMARIZE_MEMORY"
        assert "Chennai" in resp.message
        assert "Created via AI Assistant" in resp.message
        assert resp.actions[0].type == "OPEN_MEMORY"
