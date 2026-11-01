from unittest.mock import MagicMock, patch

from app.services.ai_service import AIService


def test_restored_chat_messages_only_include_the_users_media():
    client = MagicMock()
    conversations = MagicMock()
    messages = MagicMock()
    media = MagicMock()
    client.table.side_effect = {
        "ai_conversations": conversations,
        "ai_messages": messages,
        "media": media,
    }.get

    conversations.select.return_value.eq.return_value.eq.return_value.execute.return_value.data = [
        {"id": "conversation-1"}
    ]
    messages.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [
        {
            "id": "message-1",
            "role": "assistant",
            "content": "Here is the photo.",
            "related_memory_ids": ["media-owned", "media-other-user"],
            "created_at": "2026-10-31T00:00:00+00:00",
        }
    ]
    media_row = {
        "id": "media-owned",
        "filename": "family.jpg",
        "media_type": "image",
        "created_at": "2026-10-30T00:00:00+00:00",
        "owner_id": "user-1",
        "storage_path": "family.jpg",
        "url": "https://example.invalid/family.jpg",
        "file_size": 10,
    }
    media.select.return_value.in_.return_value.eq.return_value.execute.return_value.data = [
        media_row
    ]

    with patch("app.services.ai_service.get_supabase_client", return_value=client):
        restored = AIService.get_messages("user-1", "conversation-1")

    media.select.return_value.in_.return_value.eq.assert_called_once_with(
        "owner_id", "user-1"
    )
    assert [item.id for item in restored[0].related_media] == ["media-owned"]
