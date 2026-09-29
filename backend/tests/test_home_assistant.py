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
