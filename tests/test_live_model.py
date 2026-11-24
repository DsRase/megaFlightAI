"""Необязательные сценарии фактического выбора инструментов через внешний API."""
import json
import os

import pytest
from langchain_openai import ChatOpenAI

from flight_agent.agent import create_agent, respond


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
    model = ChatOpenAI(model="deepseek/deepseek-v4-flash", base_url="https://sprutdock.ru/v1", api_key=key, temperature=0, timeout=60, max_retries=1)
    answer, trace = respond(create_agent(model), query)
    print(json.dumps({"запрос": query.replace("1234567890", "***"), "ожидаемый_инструмент": tool, "фактические_вызовы": trace, "ответ": answer}, ensure_ascii=False))
    assert len(trace) == 1 and trace[0]["tool"] == tool
    assert all(trace[0]["arguments"].get(k) == v for k, v in args.items())
    assert answer["status"] == status
    if status == "ok":
        assert "задержан" in answer["message"]
    else:
        assert answer.get("booking_id") and answer.get("flight_number") == "SU123"
