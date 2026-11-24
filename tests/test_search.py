import json

from langchain_core.messages import AIMessage, ToolMessage

from flight_agent.agent import create_agent, respond
from flight_agent.data import FlightStore
from flight_agent.__main__ import print_result


def test_from_now_filters_past_departures(monkeypatch):
    from datetime import datetime
    import flight_agent.data as data

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2027, 6, 12, 10, 0)

    monkeypatch.setattr(data, "datetime", Clock)
    store = FlightStore()
    result = store.search(origin="Москва", destination="Санкт-Петербург", from_now=True)
    assert [flight["flight_number"] for flight in result["flights"]] == ["SU124"]
    assert store.search(origin="Москва", destination="Санкт-Петербург")["count"] == 3
    store.close()


class SearchModel:
    def __init__(self, arguments, reply):
        self.arguments = arguments
        self.reply = reply
        self.contexts = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.contexts.append(messages)
        if len(self.contexts) == 1:
            return AIMessage(content="", tool_calls=[{"name": "search_flights", "args": self.arguments, "id": "search-1", "type": "tool_call"}])
        return AIMessage(content=json.dumps({"status": "ok", "message": self.reply}, ensure_ascii=False))


def test_search_filters_city_cases_date_and_availability():
    store = FlightStore()
    result = store.search(origin="МОСКВЫ", destination="казань", flight_date="2027-06-10")
    assert result["count"] == 1
    assert result["flights"][0]["flight_number"] == "S7100"
    assert store.search(origin="Москва", destination="Казани", flight_date="2027-06-10", available_only=True)["count"] == 0
    assert store.search(destination="москву")["count"] == store.search(destination="Москва")["count"] > 0
    assert store.search(destination="несуществующий город")["count"] == 0
    assert store.search(flight_date="2027-02-30")["status"] == "error"
    assert store.search(flight_date="20270610")["status"] == "error"


def test_search_count_limit_and_booking_update():
    store = FlightStore()
    result = store.search(limit=2)
    assert result["count"] > 10 and len(result["flights"]) == 2 and result["truncated"]
    assert len(store.search(limit=1000)["flights"]) == 10
    assert [f["departure"] for f in result["flights"]] == sorted(f["departure"] for f in result["flights"])
    before = store.search(origin="Москва", destination="Санкт-Петербург", flight_date="2027-06-12", available_only=True)
    assert store.book("SU123", "2027-06-12", "Тест", "1234567890")["status"] == "booked"
    after = store.search(origin="Москва", destination="Санкт-Петербург", flight_date="2027-06-12", available_only=True)
    assert after["count"] == before["count"] - 1
    assert all(f["seats"] > 0 and f["status"] != "отменён" for f in after["flights"])


def test_search_tool_result_reaches_model_answer_and_cli(capsys):
    model = SearchModel({"destination": "москву", "available_only": True, "limit": 2}, "Нашлись подходящие варианты.")
    answer, trace = respond(create_agent(model), "Сколько доступных перелётов в Москву?")
    assert "search_flights" in [tool.name for tool in model.tools]
    tool_message = next(m for m in model.contexts[-1] if isinstance(m, ToolMessage))
    observed = json.loads(tool_message.content)
    assert observed["count"] > 2 and len(observed["flights"]) == 2
    assert observed["message"] == "Поиск выполнен."
    assert trace[0]["tool"] == "search_flights"
    assert trace[0]["arguments"]["available_only"] is True
    assert answer["count"] == observed["count"] and answer["flights"] == observed["flights"]
    assert str(observed["count"]) in answer["message"]
    print_result(answer, trace)
    output = capsys.readouterr().out
    assert f"Найдено рейсов: {observed['count']}" in output
    assert '"flights"' not in output and "search_flights" not in output


def test_search_result_survives_invalid_model_response():
    model = SearchModel({"flight_date": "2027-06-10"}, "")
    answer, trace = respond(create_agent(model), "Рейсы на 10 июня")
    assert answer["count"] == trace[0]["result"]["count"] > 0
    assert str(answer["count"]) in answer["message"]
    assert answer["diagnostic"]["category"] == "invalid_response"


def test_wrong_model_count_does_not_override_tool_data(capsys):
    model = SearchModel({"destination": "Казань", "limit": 2}, "Найдено 9999 рейсов.")
    answer, trace = respond(create_agent(model), "Рейсы в Казань")
    print_result(answer, trace)
    output = capsys.readouterr().out
    assert answer["count"] == trace[0]["result"]["count"] != 9999
    assert f"Найдено рейсов: {answer['count']}" in output
    assert '"flights":' not in output and "9999" not in output


def test_wrong_count_does_not_remove_booking_clarification(capsys):
    model = SearchModel({"destination": "Казань"}, "Найдено 9999 рейсов, укажите имя пассажира и паспорт.")
    answer, trace = respond(create_agent(model), "Найди рейс для бронирования")
    print_result(answer, trace)
    output = capsys.readouterr().out
    assert "9999" not in answer["message"] and "9999" not in output
    assert str(trace[0]["result"]["count"]) in answer["message"]
    assert "укажите имя пассажира" in answer["message"] and "укажите имя пассажира" in output


def test_multiple_searches_keep_each_count_and_natural_answer():
    from flight_agent.agent import prepare_answer

    records = [
        {"tool": "search_flights", "executed": True, "result": {"status": "ok", "count": count, "flights": []}}
        for count in (3, 19)
    ]
    answer = prepare_answer({"results": records, "messages": [AIMessage(content="Есть варианты. Укажите имя пассажира.")], "rounds": 1})
    assert "Есть варианты." not in answer["message"] and "Укажите имя пассажира." in answer["message"]
    assert "рейсов: 3." in answer["message"] and "рейсов: 19." in answer["message"]
    assert [item["result"]["count"] for item in answer["tool_results"]] == [3, 19]


def test_untrusted_search_claims_and_same_sentence_question():
    for claim in ("Найдено пять рейсов", "Доступны 5 подходящих рейсов", "Рейс SU123 отменён"):
        model = SearchModel({"destination": "Казань"}, claim + " — на какую дату искать?")
        answer, trace = respond(create_agent(model), "Найди рейсы")
        assert claim not in answer["message"]
        assert "на какую дату искать?" in answer["message"]
        assert f"рейсов: {trace[0]['result']['count']}" in answer["message"]


def test_multiple_searches_show_each_result_without_technical_filters(monkeypatch):
    from datetime import datetime
    from flight_agent.__main__ import format_result

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2027, 6, 12, 10, 0, tzinfo=tz)

    monkeypatch.setattr("flight_agent.data.datetime", Clock)
    arguments = [
        {"origin": "Москва", "destination": "Санкт-Петербург", "flight_date": "2027-06-12", "available_only": True, "limit": 1},
        {"origin": "Сочи", "destination": "Москва", "from_now": True},
    ]

    class TwoSearches(SearchModel):
        def invoke(self, messages):
            self.contexts.append(messages)
            if len(self.contexts) == 1:
                return AIMessage(content="", tool_calls=[
                    {"name": "search_flights", "args": args, "id": f"search-{index}", "type": "tool_call"}
                    for index, args in enumerate(arguments, 1)
                ])
            return AIMessage(content="Найдено пять рейсов — на какую дату искать?")

    answer, trace = respond(create_agent(TwoSearches({}, "")), "Найди рейсы")
    assert [r["arguments"] for r in answer["tool_results"]] == [r["arguments"] for r in trace] == arguments
    assert [r["result"] for r in answer["tool_results"]] == [r["result"] for r in trace]
    first_label = "Поиск 1 (Москва → Санкт-Петербург, 2027-06-12)"
    second_label = "Поиск 2 (Сочи → Москва)"
    first, second = answer["message"].split(first_label)[1].split(second_label)
    assert "рейсов: 2." in first and "Показаны первые 1 из 2." in first
    assert f"рейсов: {trace[1]['result']['count']}." in second
    output = format_result(answer, trace)
    first, second = output.split(first_label)[1].split(second_label)
    assert "Рейс SU123" in first and "Показано: 1." in first
    assert "Рейс SU202" in second
    for text in (answer["message"], output):
        assert "на какую дату искать?" in text
        assert all(label not in text for label in ("Условия поиска", "только доступные", "с текущего времени", "запрошенный лимит"))


def test_search_trace_redacts_sensitive_arguments():
    model = SearchModel({"destination": "паспорт 1234567890", "limit": 2}, "Укажите имя пассажира и паспорт.")
    answer, trace = respond(create_agent(model), "Найди рейсы")
    assert "1234567890" not in json.dumps([answer, trace], ensure_ascii=False)
    assert "***" in trace[0]["arguments"]["destination"]
