import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from flight_agent import web


class Model:
    def __init__(self):
        self.contexts = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.contexts.append(messages)
        return AIMessage(content=json.dumps({"status": "ok", "message": "**Здравствуйте!**", "missing_fields": [], "alternatives": []}, ensure_ascii=False))


@pytest.fixture
def client(monkeypatch, tmp_path):
    model = Model()
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("FLIGHT_DB_PATH", str(tmp_path / "flights.sqlite3"))
    monkeypatch.setattr(web, "ChatOpenAI", lambda **kwargs: model)
    with TestClient(web.app, base_url="http://localhost:8080") as client:
        yield client, model


def test_static_and_chat_context(client):
    http, model = client
    assert "chat-form" in http.get("/").text
    assert http.get("/app.js").status_code == 200
    assert http.get("/styles.css").status_code == 200
    assert http.get("/../../.env").status_code == 404
    assert http.post("/api/chat", json={"message": "Первый вопрос"}).json() == {"reply": "Здравствуйте!"}
    assert http.post("/api/chat", json={"message": "Второй вопрос"}).status_code == 200
    assert any("Первый вопрос" in str(item.content) for item in model.contexts[-1])


@pytest.mark.parametrize("body", [{}, {"message": 12}, {"message": None}])
def test_validation_does_not_echo_input(client, body):
    http, _ = client
    response = http.post("/api/chat", json=body)
    assert response.status_code == 422
    assert set(response.json()) == {"reply"}


def test_empty_and_safe_error(client, monkeypatch, caplog):
    http, _ = client
    assert http.post("/api/chat", json={"message": "  "}).status_code == 400

    def fail(*args):
        raise RuntimeError("secret 1234567890")

    monkeypatch.setattr(web, "respond", fail)
    response = http.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 500
    assert "secret" not in response.text and "1234567890" not in response.text
    assert "response_http_status=500 result_status=error" in caplog.text
    assert "'source': 'application', 'category': 'unexpected', 'code': 'runtime_error'" in caplog.text
    assert all(value not in caplog.text for value in ("secret", "1234567890", "hello"))


def test_tool_result_uses_cli_format(client, monkeypatch, caplog):
    http, _ = client
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    result = {"status": "ok", "count": 1, "flights": [{"flight_number": "SU123", "origin": "Москва", "destination": "Казань", "seats": 3}]}
    monkeypatch.setattr(web, "respond", lambda *args: ({"status": "ok", "message": "Поиск выполнен."}, [{"tool": "search_flights", "result": result}]))
    reply = http.post("/api/chat", json={"message": "Рейсы"}).json()["reply"]
    assert "Найдено рейсов: 1" in reply and "SU123" in reply and "свободных мест: 3" in reply
    started, completed = [record.getMessage() for record in caplog.records if record.name == "uvicorn.error"]
    assert started.split("request_id=")[1] == completed.split("request_id=")[1].split()[0]
    assert "response_http_status=200 result_status=ok elapsed_ms=" in completed
    assert "tools=['search_flights:ok'] diagnostics=[]" in completed
    assert all(value not in completed for value in ("Рейсы", "Поиск выполнен", "SU123", "Москва"))


def test_agent_diagnostic_logged_with_successful_http_response(client, monkeypatch, caplog):
    http, _ = client
    diagnostic = {"source": "model", "category": "rate_limit", "http_status": 429, "action": "private detail"}
    answer = {"status": "ok", "message": "Готово", "diagnostic": diagnostic}
    trace = [
        {"tool": "private tool", "arguments": {"name": "Иван"}, "result": {"status": "private status"}},
        {"tool": "search_flights", "result": {"status": "error", "diagnostic": {
            "source": "tool", "category": "unexpected", "code": "runtime_error",
        }}},
    ]
    monkeypatch.setattr(web, "respond", lambda *args: (answer, trace))
    monkeypatch.setattr(web, "format_result", lambda *args: "Готово")
    assert http.post("/api/chat", json={"message": "Введите 0123456789"}).status_code == 200
    assert "response_http_status=200 result_status=ok" in caplog.text
    assert "tools=['unknown:-', 'search_flights:error']" in caplog.text
    assert "'source': 'model', 'category': 'rate_limit', 'http_status': 429" in caplog.text
    assert "'source': 'tool', 'category': 'unexpected', 'code': 'runtime_error'" in caplog.text
    assert all(value not in caplog.text for value in ("private", "Иван", "0123456789", "Готово"))


def test_requests_are_serialized(client, monkeypatch):
    http, _ = client
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    calls = []

    def respond(graph, text, thread_id):
        calls.append((graph, thread_id))
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return {"status": "ok", "message": "Готово"}, []

    monkeypatch.setattr(web, "respond", respond)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(http.post, "/api/chat", json={"message": "1"})
        try:
            assert entered.wait(5)
            second = pool.submit(http.post, "/api/chat", json={"message": "2"})
            assert not second_entered.wait(0.2)
        finally:
            release.set()
        assert first.result().status_code == 200
        assert second.result().status_code == 200
    assert second_entered.is_set() and calls[0] == calls[1]


@pytest.mark.parametrize("host,origin,expected", [
    ("localhost:8080", None, 200),
    ("127.0.0.1:8080", "http://127.0.0.1:8080", 200),
    ("localhost:8080", "http://localhost:8080", 200),
    ("evil.example:8080", None, 403),
    ("localhost:8080", "http://evil.example:8080", 403),
    ("localhost:8080", "null", 403),
    ("localhost:8080", "http://localhost:9999", 403),
    ("localhost:9999", None, 403),
    ("localhost:8080.evil.example", None, 403),
])
def test_local_host_and_origin_required(client, host, origin, expected):
    http, _ = client
    headers = {"Host": host}
    if origin is not None:
        headers["Origin"] = origin
    assert http.post("/api/chat", headers=headers, json={"message": "Привет"}).status_code == expected
    assert http.get("/", headers=headers).status_code == expected


def test_busy_request_times_out_safely(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(web, "LOCK_TIMEOUT", 0.01)
    web.app.state.lock.acquire()
    try:
        response = http.post("/api/chat", json={"message": "1234567890"})
    finally:
        web.app.state.lock.release()
    assert response.status_code == 503
    assert "занято" in response.json()["reply"]
    assert "1234567890" not in response.text


def test_profile_results_through_web(client, monkeypatch):
    http, _ = client
    from flight_agent.data import FlightStore

    store = FlightStore()
    operations = iter([(name, getattr(store, name)(*args)) for name, args in [
        ("set_profile", ("Иван",)), ("get_profile", ()), ("delete_profile", ()), ("get_profile", ())
    ]])
    def respond(*args):
        name, result = next(operations)
        return {"status": "ok"}, [{"tool": name, "result": result}]

    monkeypatch.setattr(web, "respond", respond)
    try:
        replies = [http.post("/api/chat", json={"message": "профиль"}).json()["reply"] for _ in range(4)]
        assert replies == ["Имя сохранено: Иван.", "Ваше имя: Иван.", "Профиль удалён.", "Имя пользователя не сохранено."]
    finally:
        store.close()


def test_chat_shows_saved_passport_only_on_request(client, monkeypatch):
    http, model = client
    store = web.app.state.store
    passport = "0123456789"
    store.set_profile("Иван")
    assert store.set_profile_passport(passport, "Иван")
    responses = iter([
        AIMessage(content="", tool_calls=[{"name": "show_saved_passport", "args": {}, "id": "show"}]),
        AIMessage(content="Готово."),
        AIMessage(content='{"status":"ok","message":"Здравствуйте!","missing_fields":[],"display_saved_passport":"Иван"}'),
    ])

    def invoke(messages):
        model.contexts.append(messages)
        return next(responses)

    monkeypatch.setattr(model, "invoke", invoke)
    reply = http.post("/api/chat", json={"message": "Покажи номер моего паспорта"}).json()["reply"]
    assert passport in reply
    reply = http.post("/api/chat", json={"message": "Привет"}).json()["reply"]
    assert reply == "Здравствуйте!"
    assert all(passport not in str(messages) for messages in model.contexts)
    config = {"configurable": {"thread_id": web.app.state.thread_id}}
    assert all(passport not in repr(checkpoint) for checkpoint in web.app.state.graph.checkpointer.list(config))


def test_clear_changes_thread_and_preserves_database(client):
    http, model = client
    from flight_agent.data import FlightStore
    import os

    store = FlightStore(os.environ["FLIGHT_DB_PATH"])
    try:
        store.set_profile("Иван")
        booking = store.book("SU123", "2027-06-10", "Иван", "1234567890")
        http.post("/api/chat", json={"message": "Старый вопрос"})
        previous = web.app.state.thread_id
        assert http.post("/api/chat/clear").status_code == 200
        assert list(web.app.state.graph.checkpointer.list({"configurable": {"thread_id": previous}})) == []
        assert web.app.state.thread_id != previous
        http.post("/api/chat", json={"message": "Новый вопрос"})
        assert not any("Старый вопрос" in str(m.content) for m in model.contexts[-1])
        assert store.get_profile()["name"] == "Иван"
        assert store.list_bookings()["bookings"][0]["booking_id"] == booking["booking_id"]
    finally:
        store.close()


def test_clear_uses_chat_lock(client, monkeypatch):
    http, _ = client
    monkeypatch.setattr(web, "LOCK_TIMEOUT", 0.01)
    previous = web.app.state.thread_id
    web.app.state.lock.acquire()
    try:
        assert http.post("/api/chat/clear").status_code == 503
        assert web.app.state.thread_id == previous
    finally:
        web.app.state.lock.release()


def test_message_size_limit(client):
    http, model = client
    assert http.post("/api/chat", json={"message": "а" * (web.MAX_MESSAGE_CHARS + 1)}).status_code == 400
    assert not model.contexts
