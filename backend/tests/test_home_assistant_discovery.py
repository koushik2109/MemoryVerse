from app.services.home_assistant_service import home_assistant_service


def test_create_journal_of_place_routes_to_grounded_media_discovery():
    result = home_assistant_service.process_command(
        user_id="user-1",
        query="create journal of Shimla",
    )

    assert result.intent == "CREATE_MEMORY_DISCOVERY"
    assert result.actions
    action = result.actions[0]
    assert action.type == "CREATE_MEMORY"
    assert action.data == {
        "initial_prompt": "Create journal of Shimla",
        "auto_start_discovery": True,
    }
    assert "scan" in result.message.lower()
