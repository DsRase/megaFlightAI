import json

import pytest
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
    class BookingModel(Model):
        def invoke(self, messages):
            self.contexts.append(messages)
            human = next(str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage))
            if len(self.contexts) == 1:
                return AIMessage(content="", tool_calls=[
                    {"name": "prepare_booking", "args": {"origin": "Сочи", "destination": "Москва", "own": True}, "id": "p", "type": "tool_call"},
                    {"name": "get_profile", "args": {}, "id": "g", "type": "tool_call"}])
            if "PVT_" in human and not isinstance(messages[-1], ToolMessage):
                import re
                marker = re.search(r"PVT_[a-f0-9]{24}", human).group()
                return call("book_flight", {"flight_number": "WRONG", "flight_date": "2027-01-01", "passenger_name": "Вадим Гутник", "passport": marker})
            raise RuntimeError("Текст модели после инструмента не нужен")

    model = BookingModel([])
    store = FlightStore()
    graph = create_agent(model, store)
    answer, trace = respond(graph, "Заведи на меня ближайший рейс из сочи в москву")
    assert "Ближайший рейс:" in answer["message"]
    flight = trace[0]["result"]["flight"]
    assert flight["flight_number"] in answer["message"]
    assert answer["missing_fields"] == ["passenger_name", "passport"]
    assert "count" not in trace[0]["result"]
    assert "Пришлите имя пассажира и 10 цифр паспорта." in answer["message"]
    output = format_result(answer, trace)
    assert output.count(flight["flight_number"]) == 1
    assert "Пришлите имя пассажира и 10 цифр паспорта." in output
    assert all(text not in output for text in ("Имя пользователя не сохранено", "Условия поиска", "Не удалось"))
    booked, booking_trace = respond(graph, "Вадим Гутник, паспорт 1234567890")
    assert booked["status"] == "booked" and booked["booking_id"] == "DEMO-0001"
    assert (booked["flight_number"], booked["flight_date"]) == (flight["flight_number"], flight["departure"][:10])
    assert store.get_profile()["name"] == "Вадим Гутник"
    assert all(text not in format_result(booked, booking_trace) for text in ("Имя сохранено", "Не удалось", "Попробуйте"))
    assert "1234567890" not in json.dumps([answer, trace, booked, booking_trace, graph.get_state({"configurable": {"thread_id": "cli"}}).values], default=str)
    assert "1234567890" not in str(model.contexts)
    assert {"list_bookings", "cancel_booking", "update_booking"} <= model.tools
    store.close()


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


@pytest.mark.parametrize(("reply", "diagnostic_code"), [
    ("Ваше имя Иван. У вас одна активная бронь.", None),
    ('{"status":', "invalid_format"),
    ("", "empty_content"),
])
def test_profile_and_bookings_remain_visible_after_final_model_reply(reply, diagnostic_code):
    store = FlightStore()
    store.set_profile("Иван")
    booking = store.book("SU123", "2027-06-10", "Иван", "1234567890")
    model = Model([call("get_profile", {}, "profile"), call("list_bookings", {}, "bookings"), AIMessage(content=reply)])
    answer, trace = respond(create_agent(model, store), "Привет, кто я и какие у меня брони?")
    text = format_result(answer, trace)
    assert "Ваше имя: Иван." in text and booking["booking_id"] in text
    assert "Не удалось полностью обработать запрос" not in text and "Попробуйте" not in text
    if diagnostic_code:
        assert answer["diagnostic"]["source"] == "model"
        assert answer["diagnostic"]["code"] == diagnostic_code
    else:
        assert "diagnostic" not in answer
    store.close()


def test_model_reply_failure_keeps_empty_booking_list_visible():
    store = FlightStore()
    model = Model([call("list_bookings", {}), AIMessage(content="")])
    answer, trace = respond(create_agent(model, store), "Какие у меня брони?")
    assert format_result(answer, trace) == "Активных броней нет."
    assert answer["diagnostic"]["code"] == "empty_content"
    store.close()


def test_model_reply_failure_preserves_domain_error():
    store = FlightStore()
    model = Model([
        call("list_bookings", {}, "bookings"),
        call("flight_status", {"flight_number": "SU123", "flight_date": "ошибка"}, "flight"),
        AIMessage(content=""),
    ])
    answer, trace = respond(create_agent(model, store), "Мои брони и статус рейса")
    text = format_result(answer, trace)
    assert "Активных броней нет." in text
    assert "Дата должна быть в формате YYYY-MM-DD." in text
    assert "Не удалось полностью обработать запрос" not in text
    assert answer["diagnostic"]["code"] == "empty_content"
    store.close()


def test_profile_and_bookings_hide_unrelated_tool_failure():
    class UnexpectedError(Exception):
        pass

    class Store(FlightStore):
        def status(self, *args):
            raise UnexpectedError("private detail")

    store = Store()
    try:
        store.set_profile("Иван")
        booking = store.book("SU123", "2027-06-10", "Иван", "1234567890")
        model = Model([
            call("flight_status", {"flight_number": "SU123"}, "flight"),
            call("list_bookings", {}, "bookings"),
            call("get_profile", {}, "profile"),
            AIMessage(content="Ваше имя Иван. У вас одна активная бронь."),
        ])
        answer, trace = respond(create_agent(model, store), "Кто я и какие на мне брони есть?")
        text = format_result(answer, trace)
    finally:
        store.close()

    assert [(item["tool"], item["result"]["status"]) for item in trace] == [
        ("flight_status", "error"), ("list_bookings", "ok"), ("get_profile", "ok"),
    ]
    assert answer["diagnostic"]["code"] == trace[0]["result"]["diagnostic"]["code"] == "other_error"
    assert text.startswith(f"Бронь: {booking['booking_id']}\nРейс SU123")
    assert text.endswith("Ваше имя: Иван.")
    assert "private detail" not in str((answer, trace))
    assert all(message not in text for message in ("Не удалось", "Данные сейчас недоступны.", "Повторите запрос"))


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


def test_successful_booking_does_not_require_model_restatement():
    class FailingAfterTool(Model):
        def invoke(self, messages):
            if self.contexts:
                raise RuntimeError("model unavailable")
            self.contexts.append(messages)
            import re
            marker = re.search(r"PVT_[a-f0-9]{24}", str(messages[-1].content)).group()
            return AIMessage(content="", tool_calls=[
                {"name": "book_flight", "args": {"flight_number": "SU123", "flight_date": "2027-06-10", "passenger_name": "Иван", "passport": marker}, "id": "booking", "type": "tool_call"},
                {"name": "search_flights", "args": {"flight_date": "ошибка"}, "id": "search", "type": "tool_call"},
            ])

    store = FlightStore()
    model = FailingAfterTool([])
    answer, trace = respond(create_agent(model, store), "Иван 1234567890")
    assert answer["status"] == "booked"
    output = format_result(answer, trace)
    assert "Дата должна быть в формате YYYY-MM-DD." in output
    assert "Не удалось" not in output
    assert len(trace) == 2 and len(model.contexts) == 1
    store.close()


def test_prepare_booking_reports_no_matching_flight():
    model = Model([call("prepare_booking", {"origin": "Несуществующий город", "destination": "Москва"})])
    answer, trace = respond(create_agent(model), "Забронируй ближайший рейс")
    assert trace[0]["tool"] == "prepare_booking"
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


def test_own_booking_context_stops_when_model_clears_selection():
    for followup in ("не бронируй мне", "нет", "забронируй на имя Пётр"):
        store = FlightStore()
        store.set_profile("Иван")
        store.set_profile_passport("1234567890", "Иван")
        model = Model([call("prepare_booking", {"origin": "Сочи", "destination": "Москва", "own": True}),
                       call("clear_pending_booking", {}), AIMessage(content="Хорошо"), AIMessage(content="Хорошо")])
        graph = create_agent(model, store)
        respond(graph, "Забронируй на себя ближайший рейс из Сочи в Москву")
        respond(graph, followup)
        assert any("Для своей брони доступно имя профиля" in str(m.content) for m in model.contexts[-2] if m.type == "system")
        respond(graph, "Что дальше?")
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
            if len(self.contexts) == 1:
                return call("prepare_booking", {"origin": "Сочи", "destination": "Москва", "own": True})
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
