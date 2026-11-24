"""Необязательные сценарии фактического выбора инструментов через внешний API."""
import json
import os
from datetime import datetime

import pytest
from langchain_openai import ChatOpenAI

from flight_agent import BASE_URL, MODEL
from flight_agent.agent import create_agent, redact, respond
from flight_agent.data import FlightStore


@pytest.mark.skipif(os.environ.get("FLIGHT_LIVE_TESTS") != "1", reason="Задайте FLIGHT_LIVE_TESTS=1 для внешнего API")
@pytest.mark.parametrize("scenario", ["status", "booking"])
def test_live_tool_selection(scenario):
    if scenario == "status":
        query, tool, args, status = "Какой статус рейса SU123 2027-06-12?", "flight_status", {"flight_number": "SU123", "flight_date": "2027-06-12"}, "ok"
    else:
        query, tool, args, status = "Забронируй рейс SU123 2027-06-12 на Иван Тестов, паспорт 1234567890.", "book_flight", {"flight_number": "SU123", "flight_date": "2027-06-12", "passenger_name": "Иван Тестов", "passport": "***"}, "booked"
    key = os.environ.get("LLM_API_KEY")
    if not key:
        pytest.skip("Не задан LLM_API_KEY")
    model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=key, temperature=None, timeout=60, max_retries=1)
    answer, trace = respond(create_agent(model), query)
    print(json.dumps({"запрос": redact(query), "ожидаемый_инструмент": tool, "фактические_вызовы": trace, "ответ": answer}, ensure_ascii=False))
    assert len(trace) == 1 and trace[0]["tool"] == tool
    assert all(trace[0]["arguments"].get(k) == v for k, v in args.items())
    assert answer["status"] == status
    if status == "ok":
        assert "задержан" in answer["message"]
    else:
        assert answer.get("booking_id") and answer.get("flight_number") == "SU123"


@pytest.mark.skipif(os.environ.get("FLIGHT_LIVE_TESTS") != "1", reason="Задайте FLIGHT_LIVE_TESTS=1 для внешнего API")
def test_live_multi_turn_booking(monkeypatch):
    key = os.environ.get("LLM_API_KEY")
    if not key:
        pytest.skip("Не задан LLM_API_KEY")

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2027, 6, 1, tzinfo=tz)

    monkeypatch.setattr("flight_agent.data.datetime", FixedDateTime)
    store = FlightStore(":memory:")
    try:
        store.set_profile("Иван Тестов")
        model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=key, temperature=None, timeout=60, max_retries=1)
        graph = create_agent(model, store)
        turns = ["заведи на меня бронь на ближайший ресь из сочи в москву",
                 "так ты заведешь или нет?", "ты же знаешь, я ведь вводил ранее",
                 "а паспорт 1234567890"]
        traces = []
        for query in turns:
            answer, trace = respond(graph, query, "live-multi-turn")
            traces.extend(trace)
            print(json.dumps({"запрос": redact(query), "вызовы": trace, "ответ": answer}, ensure_ascii=False))
        assert any(call["tool"] == "search_flights" for call in traces)
        assert answer["status"] == "booked" and answer.get("booking_id")
        assert any(call["tool"] == "book_flight" for call in traces)
    finally:
        store.close()
