import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from flight_agent.agent import compact_history, create_agent, respond, safe_result
from flight_agent.data import FlightStore
from flight_agent.__main__ import format_result


class Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.contexts = []

    def bind_tools(self, tools):
        self.tools = {tool.name for tool in tools}
        return self

    def invoke(self, messages):
        self.contexts.append(messages)
        return next(self.replies)


def call(name, args, ident="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident, "type": "tool_call"}])


def test_search_answer_and_selection_survive_next_turn():
    text = "Ближайший доступный рейс SU202, 2027-06-01. Укажите имя пассажира и паспорт."
    model = Model([call("search_flights", {"origin": "Сочи", "destination": "Москва", "available_only": True, "from_now": True, "limit": 1}),
                   AIMessage(content=text), AIMessage(content="Укажите паспорт.")])
    graph = create_agent(model)
    answer, trace = respond(graph, "забронировать ближайший из сочи в москву")
    assert "Ближайший рейс:" in answer["message"]
    assert trace[0]["result"]["flights"][0]["flight_number"] in answer["message"]
    assert "Пришлите имя пассажира и 10 цифр паспорта." in answer["message"]
    assert format_result(answer, trace).count(trace[0]["result"]["flights"][0]["flight_number"]) == 1
    assert "Пришлите имя пассажира и 10 цифр паспорта." in format_result(answer, trace)
    respond(graph, "Пассажир Иван")
    assert any("Пришлите имя пассажира и 10 цифр паспорта." in str(m.content) for m in model.contexts[-1])
    assert {"list_bookings", "cancel_booking", "update_booking"} <= model.tools


def test_management_tools_and_safe_formatting():
    store = FlightStore()
    ident = store.book("SU123", "2027-06-10", "Иван", "1234567890")["booking_id"]
    model = Model([call("list_bookings", {}), AIMessage(content="Ваши активные брони."),
                   call("update_booking", {"booking_id": ident, "flight_number": "SU124", "flight_date": "2027-06-12"}), AIMessage(content="Готово."),
                   call("cancel_booking", {"booking_id": ident}), AIMessage(content="Готово.")])
    graph = create_agent(model, store)
    try:
        listed, trace = respond(graph, "Мои брони")
        assert listed["bookings"][0]["flight_status"] == "вовремя"
        updated, update_trace = respond(graph, "Перенеси бронь")
        cancelled, cancel_trace = respond(graph, "Отмени бронь")
        assert updated["status"] == "updated" and "перенесена" in format_result(updated, update_trace)
        assert cancelled["status"] == "cancelled" and "отменена" in format_result(cancelled, cancel_trace)
        assert "1234567890" not in json.dumps([listed, trace, updated, update_trace, cancelled, cancel_trace])
        assert safe_result({"bookings": [{"booking_id": ident, "passport_fingerprint": "secret"}]}) == {"bookings": [{"booking_id": ident}]}
    finally:
        store.close()


def test_compaction_keeps_whole_tool_turns():
    messages = []
    for index in range(8):
        messages.extend([HumanMessage(content=f"запрос {index}", id=f"h{index}"),
                         call("list_bookings", {}, f"c{index}").model_copy(update={"id": f"a{index}"}),
                         ToolMessage(content="{}", tool_call_id=f"c{index}", id=f"t{index}"),
                         AIMessage(content="ответ", id=f"r{index}")])
    kept, removed = compact_history(messages, max_chars=24000, recent_turns=3)
    assert len(removed) == 20 and len(kept) == 12
    assert kept[0].content == "запрос 5"
    for index in range(0, len(kept), 4):
        assert kept[index + 1].tool_calls[0]["id"] == kept[index + 2].tool_call_id


def test_compaction_in_graph_keeps_recent_turns_and_profile(tmp_path):
    store = FlightStore(tmp_path / "flights.db")
    store.set_profile("Иван")
    model = Model([AIMessage(content="Ответ") for _ in range(9)])
    graph = create_agent(model, store)
    try:
        for index in range(9):
            respond(graph, f"Запрос {index}: " + "а" * 5000)
        assert len(model.contexts) == 9
        assert not any("Запрос 0:" in str(m.content) for m in model.contexts[-1])
        assert any("Запрос 8:" in str(m.content) for m in model.contexts[-1])
        assert store.get_profile()["name"] == "Иван"
    finally:
        store.close()


def test_short_turns_are_compacted_and_checkpoint_versions_pruned():
    model = Model([AIMessage(content="Ответ") for _ in range(10)])
    graph = create_agent(model)
    for index in range(10):
        respond(graph, f"Вопрос {index}", "bounded")
    assert not any("Вопрос 0" in str(m.content) for m in model.contexts[-1])
    assert len(list(graph.checkpointer.list({"configurable": {"thread_id": "bounded"}}))) == 1


def test_booking_list_output_includes_data_even_with_natural_summary():
    store = FlightStore()
    booking = store.book("SU123", "2027-06-10", "Иван", "1234567890")
    model = Model([call("list_bookings", {}), AIMessage(content="Ваши активные брони.")])
    answer, trace = respond(create_agent(model, store), "Мои брони")
    text = format_result(answer, trace)
    assert "Ваши активные брони" in text and booking["booking_id"] in text and "SU123" in text
    store.close()


def test_multiple_management_results_include_success_and_failure():
    from flight_agent.agent import prepare_answer
    records = [
        {"tool": "cancel_booking", "executed": True, "result": {"status": "cancelled", "booking_id": "DEMO-0001", "already_cancelled": False}},
        {"tool": "cancel_booking", "executed": True, "result": {"status": "not_found", "message": "Бронь не найдена."}},
    ]
    answer = prepare_answer({"results": records, "messages": [AIMessage(content="Готово")], "rounds": 1})
    output = format_result(answer, records)
    assert "DEMO-0001" in output and "Бронь не найдена" in output
    assert len(answer["tool_results"]) == 2


def test_passport_mask_preserves_questions_and_dates():
    from flight_agent.agent import redact
    assert redact("паспортные данные; а паспорт какой; 2027-06-12 12:30; найдено 123 рейса") == "паспортные данные; а паспорт какой; 2027-06-12 12:30; найдено 123 рейса"
    assert redact("паспорт 987; passport ABC/DEF; 12 34 567890") == "паспорт ***; passport ***; ***"


def test_repeated_booking_rejection_not_printed():
    from flight_agent.agent import prepare_answer
    records = [
        {"tool": "book_flight", "executed": True, "result": {"status": "booked", "message": "Бронь создана."}},
        {"tool": "book_flight", "executed": False, "result": {"status": "error", "message": "Допускается один вызов бронирования за ход."}},
    ]
    answer = prepare_answer({"results": records, "messages": [AIMessage(content="Готово")], "rounds": 1})
    assert answer["status"] == "booked"
    assert "Допускается" not in format_result(answer, records)


def test_nearest_empty_and_model_search_arguments_are_corrected():
    model = Model([call("search_flights", {"origin": "Несуществующий город"}), AIMessage(content="")])
    answer, trace = respond(create_agent(model), "Забронируй ближайший рейс")
    assert trace[0]["arguments"]["limit"] == 1
    assert trace[0]["arguments"]["from_now"] is True
    assert "Нет подходящих рейсов" in answer["message"]
    assert "Нет подходящих рейсов" in format_result(answer, trace)


def test_profile_passport_requires_active_own_booking():
    store = FlightStore()
    store.set_profile("Иван")
    store.set_profile_passport("1234567890", "Иван")
    args = {"flight_number": "SU123", "flight_date": "2027-06-10", "passenger_name": "Иван", "passport": "PVT_PROFILE"}
    model = Model([AIMessage(content="Хорошо"), call("book_flight", args), AIMessage(content="Готово")])
    graph = create_agent(model, store)
    respond(graph, "Покажи мне рейс")
    answer, _ = respond(graph, "Ну как?")
    assert answer["status"] != "booked"
    assert not store.list_bookings()["bookings"]
    store.close()


def test_own_booking_context_stops_on_refusal_or_new_passenger():
    for followup in ("не бронируй мне", "нет", "забронируй на имя Пётр", "а паспорт сохранён?"):
        store = FlightStore()
        store.set_profile("Иван")
        store.set_profile_passport("1234567890", "Иван")
        model = Model([AIMessage(content="Хорошо"), AIMessage(content="Хорошо")])
        graph = create_agent(model, store)
        respond(graph, "Забронируй на себя ближайший рейс")
        assert any("PVT_PROFILE" in str(m.content) for m in model.contexts[-1] if m.type == "system")
        respond(graph, followup)
        assert not any("PVT_PROFILE" in str(m.content) for m in model.contexts[-1] if m.type == "system" and "Для своей брони" in str(m.content))
        store.close()


def test_old_booking_does_not_turn_nearest_search_into_booking():
    model = Model([AIMessage(content="Хорошо"), call("search_flights", {"limit": 1, "from_now": True, "available_only": True}), AIMessage(content="Нашёл рейс")])
    graph = create_agent(model)
    respond(graph, "Забронируй ближайший рейс")
    answer, _ = respond(graph, "Покажи ближайший рейс")
    assert "паспорт" not in answer["message"].lower()


def test_other_passport_does_not_replace_profile():
    store = FlightStore()
    store.set_profile("Иван")
    store.set_profile_passport("1234567890", "Иван")
    model = Model([AIMessage(content="Хорошо"), AIMessage(content="Хорошо")])
    graph = create_agent(model, store)
    respond(graph, "Заведи бронь на меня")
    respond(graph, "паспорт друга 0987654321")
    assert store.get_profile_passport("Иван") == "1234567890"
    store.close()


def test_repeated_booking_after_refusal_has_one_reason():
    from flight_agent.agent import prepare_answer
    records = [
        {"tool": "book_flight", "executed": False, "result": {"status": "missing_fields", "message": "Укажите паспорт повторно: данные предыдущего хода не сохраняются.", "missing_fields": ["passport"]}},
        {"tool": "book_flight", "executed": False, "result": {"status": "error", "message": "Допускается один вызов бронирования за ход."}},
    ]
    answer = prepare_answer({"results": records, "messages": [AIMessage(content="")], "rounds": 1})
    assert "Допускается" not in answer["message"]
    assert "Допускается" not in format_result(answer, records)


def test_active_own_booking_accepts_passport_on_followup_and_ends_after_success():
    import re

    class BookingModel(Model):
        def invoke(self, messages):
            self.contexts.append(messages)
            human = next(str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage))
            if "PVT_" in human and not any(isinstance(m, ToolMessage) for m in messages[messages.index(next(m for m in reversed(messages) if isinstance(m, HumanMessage))) + 1:]):
                marker = re.search(r"PVT_[a-f0-9]{24}", human).group()
                return call("book_flight", {"flight_number": "SU123", "flight_date": "2027-06-10", "passenger_name": "Иван", "passport": marker})
            return AIMessage(content="Уточните данные.")

    store = FlightStore()
    store.set_profile("Иван")
    model = BookingModel([])
    graph = create_agent(model, store)
    respond(graph, "Заведи на меня бронь на ближайший рейс")
    respond(graph, "Так ты заведешь или нет?")
    respond(graph, "Ты же знаешь, я ведь вводил ранее")
    answer, _ = respond(graph, "1234567890")
    assert answer["status"] == "booked"
    assert store.get_profile_passport("Иван") == "1234567890"
    respond(graph, "Что дальше?")
    assert not any("PVT_PROFILE" in str(m.content) for m in model.contexts[-1] if m.type == "system" and "Для своей брони" in str(m.content))
    store.close()
