"""Локальные OpenAI-compatible ответы проходят через настоящий адаптер и граф."""
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from flight_agent.agent import create_agent, respond
from flight_agent.__main__ import main


def mock_model(messages):
    requests = []
    replies = iter(messages)

    def handle(request):
        requests.append(json.loads(request.content))
        message = next(replies)
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
            "model": "test", "choices": [{"index": 0, "message": message,
                                           "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return ChatOpenAI(model="test", api_key="test-key", base_url="https://example.invalid/v1", http_client=client, max_retries=0), requests


def test_two_turns_through_openai_graph_and_cli(monkeypatch, capsys):
    model, requests = mock_model([
        {"role": "assistant", "content": "Какой вопрос вас интересует?"},
        {"role": "assistant", "content": "Вы хотите узнать статус, расписание или забронировать перелёт? Укажите номер рейса и дату."},
    ])
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr("flight_agent.__main__.ChatOpenAI", lambda **kwargs: model)
    inputs = iter(["Какой вопрос?", "какой сейчас перелет", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    assert main() == 0
    output = capsys.readouterr().out
    assert "Инструменты:" not in output
    assert "Какой вопрос вас интересует?" in output
    assert "статус, расписание или забронировать перелёт?" in output
    assert "номер рейса и дату" in output
    assert len(requests) == 2
    assert any("Какой вопрос вас интересует?" in str(m.get("content")) for m in requests[1]["messages"])


@pytest.mark.parametrize("content", [
    '```json\n{"status":"ok","message":"Чем я могу вам помочь?","missing_fields":[],"alternatives":[]}\n```',
    [{"type": "text", "text": "Чем я могу вам помочь?"}],
])
def test_openai_markdown_and_content_blocks(content):
    model, _ = mock_model([{"role": "assistant", "content": content}])
    answer, trace = respond(create_agent(model), "Привет")
    assert trace == []
    assert answer["message"] == "Чем я могу вам помочь?"


@pytest.mark.parametrize("content,code", [(None, "empty_content"), ("", "empty_content"), ('{"status":', "invalid_format")])
def test_invalid_model_response_has_specific_diagnostic(content, code, capsys):
    from flight_agent.__main__ import print_result

    model, _ = mock_model([{"role": "assistant", "content": content, "reasoning_content": "private reasoning"}])
    answer, trace = respond(create_agent(model), "Вопрос")
    assert answer["diagnostic"]["category"] == "invalid_response"
    assert answer["diagnostic"]["code"] == code
    assert "private reasoning" not in str(answer)
    print_result(answer, trace)
    output = capsys.readouterr().out
    assert output.strip() == "Данные сейчас недоступны."
    assert "invalid_response" not in output


def test_real_tool_call_response_and_untrusted_final_text():
    model, requests = mock_model([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "flight_status", "arguments": '{"flight_number":"SU123","flight_date":"2027-06-12"}'}}]},
        {"role": "assistant", "content": "Рейс летит вовремя, бронь подтверждена."},
    ])
    answer, trace = respond(create_agent(model), "Статус SU123 на 2027-06-12")
    assert trace[0]["tool"] == "flight_status"
    assert answer["flight"]["status"] == "задержан"
    assert "задержан" in answer["message"]
    assert "бронь подтверждена" not in answer["message"]
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "call_1" for m in requests[1]["messages"])


def test_invalid_tool_arguments_are_model_protocol_error():
    model, _ = mock_model([{"role": "assistant", "content": None, "tool_calls": [{"id": "call_bad", "type": "function", "function": {"name": "flight_status", "arguments": "{"}}]}])
    answer, trace = respond(create_agent(model), "Статус рейса")
    assert not trace
    assert answer["diagnostic"]["category"] == "invalid_response"
    assert answer["diagnostic"]["code"] == "invalid_tool_call"


def test_openai_booking_uses_turn_vault_and_returns_tool_result():
    class BookingTransport:
        def __init__(self):
            self.requests = []

        def handle(self, request):
            body = json.loads(request.content)
            self.requests.append(body)
            if len(self.requests) == 1:
                marker = body["messages"][-1]["content"].split()[-1]
                args = {"flight_number": "SU123", "flight_date": "2027-06-12", "passenger_name": "Иван Тестов", "passport": marker}
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": "booking", "type": "function", "function": {"name": "book_flight", "arguments": json.dumps(args, ensure_ascii=False)}}]}
            else:
                message = {"role": "assistant", "content": "Готово"}
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1, "model": "test", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]})

    transport = BookingTransport()
    client = httpx.Client(transport=httpx.MockTransport(transport.handle))
    model = ChatOpenAI(model="test", api_key="test-key", base_url="https://example.invalid/v1", http_client=client, max_retries=0)
    graph = create_agent(model)
    answer, trace = respond(graph, "Забронируй SU123 на 2027-06-12, Иван Тестов, паспорт 1234567890")
    assert answer["status"] == "booked"
    assert trace[0]["tool"] == "book_flight"
    assert "1234567890" not in str([answer, trace, transport.requests, graph.get_state({"configurable": {"thread_id": "cli"}}).values])
