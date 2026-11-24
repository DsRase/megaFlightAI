import io
import threading
from contextlib import nullcontext

import pytest

from flight_agent import __main__ as cli
from flight_agent.data import FLIGHTS


class TtyStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.updated = threading.Event()

    def isatty(self):
        return True

    def write(self, text):
        result = super().write(text)
        if "- " in text:
            self.updated.set()
        return result


def setup_cli(monkeypatch, inputs, respond, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("FLIGHT_DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setattr(cli, "ChatOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(cli, "create_agent", lambda model, store: object())
    monkeypatch.setattr(cli, "terminal_input", nullcontext)
    monkeypatch.setattr(cli, "respond", respond)
    values = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))


def test_welcome_examples_match_demo_flights_and_exit(monkeypatch, capsys, tmp_path):
    setup_cli(monkeypatch, ["/quit"], lambda *args: pytest.fail("unexpected request"), tmp_path)
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "Учебный агент" in output and "реальный билет не оформляется" in output
    assert "вымышленные данные" in output and "10 цифр" in output
    assert all(command in output for command in ("/exit", "/quit", "Ctrl-D", "Ctrl-C"))
    assert any(f[0] == "SU123" and f[4].startswith("2027-06-10") for f in FLIGHTS)
    assert any(f[0] == "SU124" and f[4].startswith("2027-06-12") for f in FLIGHTS)
    assert "SU123 на 2027-06-10" in output
    assert "SU124 на 2027-06-12" in output


def test_tty_indicator_animates_with_one_random_phrase_and_clears(monkeypatch):
    stream = TtyStream()
    monkeypatch.setattr(cli.sys, "stderr", stream)
    monkeypatch.setattr(cli.random, "choice", lambda messages: messages[0])
    with cli.waiting_indicator():
        assert stream.updated.wait(3), "indicator did not advance"
    output = stream.getvalue()
    assert len(cli.WAITING_MESSAGES) >= 10
    assert "/ " in output and "- " in output
    assert cli.WAITING_MESSAGES[0] in output
    assert all(message not in output for message in cli.WAITING_MESSAGES[1:])
    assert output.endswith("\r" + " " * (len(cli.WAITING_MESSAGES[0]) + 2) + "\r")


@pytest.mark.parametrize("error", [RuntimeError("secret 1234567890"), KeyboardInterrupt()])
def test_indicator_clears_on_error_and_interrupt(monkeypatch, error):
    stream = TtyStream()
    monkeypatch.setattr(cli.sys, "stderr", stream)
    with pytest.raises(type(error)):
        with cli.waiting_indicator():
            raise error
    output = stream.getvalue()
    assert output.endswith("\r")
    assert "secret" not in output and "1234567890" not in output


def test_non_tty_fallback_and_safe_error_before_next_trace(monkeypatch, capsys, tmp_path):
    calls = []

    def respond(graph, text, thread_id):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("secret 1234567890")
        return {"status": "ok", "message": "Уточните, пожалуйста, ваш запрос."}, []

    setup_cli(monkeypatch, ["private 1234567890", "другой вопрос", "/exit"], respond, tmp_path)
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "\r" not in captured.err
    assert "Данные сейчас недоступны." in captured.out
    assert "Диагностика:" not in captured.out and "Инструменты:" not in captured.out
    assert "secret" not in captured.out + captured.err
    assert "1234567890" not in captured.out + captured.err


def test_ctrl_c_during_request_exits_after_indicator_cleanup(monkeypatch, tmp_path):
    stream = TtyStream()
    monkeypatch.setattr(cli.sys, "stderr", stream)

    def interrupted(*args):
        raise KeyboardInterrupt

    setup_cli(monkeypatch, ["вопрос"], interrupted, tmp_path)
    assert cli.main() == 0
    assert stream.getvalue().endswith("\r")


def test_tty_indicator_is_cleared_before_trace(monkeypatch, tmp_path):
    stream = TtyStream()
    monkeypatch.setattr(cli.sys, "stderr", stream)

    def respond(*args):
        assert stream.updated.wait(3)
        return {"status": "ok"}, [{"tool": "flight_status"}]

    def print_result(answer, trace):
        stream.write("TRACE\n")

    monkeypatch.setattr(cli, "print_result", print_result)
    setup_cli(monkeypatch, ["статус", "/exit"], respond, tmp_path)
    assert cli.main() == 0
    output = stream.getvalue()
    assert any(message in output for message in cli.WAITING_MESSAGES)
    assert output.endswith("\rTRACE\n")


def test_interrupted_result_is_printed_before_exit(monkeypatch, capsys, tmp_path):
    result = {"status": "booked", "message": "Демобронь создана.", "booking_id": "DEMO-0001"}
    setup_cli(monkeypatch, ["бронь"], lambda *args: ({**result, "_interrupted": True}, [{"tool": "book_flight", "executed": True, "result": result}]), tmp_path)
    assert cli.main() == 0
    assert "DEMO-0001" in capsys.readouterr().out


def test_search_and_alternatives_are_readable(capsys):
    flight = {"flight_number": "SU123", "origin": "Москва", "destination": "Сочи", "departure": "2027-06-10T09:00", "status": "вовремя", "seats": 2}
    cli.print_result({}, [{"tool": "search_flights", "executed": True, "result": {"status": "ok", "count": 12, "truncated": True, "flights": [flight]}}])
    output = capsys.readouterr().out
    for value in ("Найдено рейсов: 12", "Показано: 1", "SU123", "Москва — Сочи", "2027-06-10 09:00", "вовремя", "мест: 2"):
        assert value in output
    cli.print_result({}, [{"tool": "book_flight", "executed": True, "result": {"status": "not_found", "message": "На эту дату рейса нет.", "alternatives": [flight]}}])
    output = capsys.readouterr().out
    assert "альтернатив: 1" in output and "SU123" in output
    assert "{" not in output and "book_flight" not in output


def test_markdown_cleanup_preserves_identifiers_and_redacts_first(capsys):
    cli.print_result({"message": "# Заголовок\n**Важно** и `код`\n- пункт\nfoo_bar и A*B\nключ: sk-secret"}, [])
    output = capsys.readouterr().out
    assert "Заголовок" in output and "Важно и код" in output and "• пункт" in output
    assert "foo_bar" in output and "A*B" in output
    assert "sk-secret" not in output and "**Важно**" not in output


def test_common_markdown_cleanup():
    text = "> Цитата\n- [x] ~~готово~~\n![Схема](https://example.org/a)\nЗаголовок\n===\n---"
    result = cli.plain_text(text)
    assert result == "Цитата\n• готово\nСхема (https://example.org/a)\nЗаголовок"


def test_unconfirmed_booking_is_not_presented_as_success(capsys):
    cli.print_result({"status": "booked", "message": "Демобронь создана.", "booking_id": "DEMO-0001"}, [])
    output = capsys.readouterr().out
    assert "Проверьте состояние броней" in output and "DEMO-0001" not in output


def test_booking_remains_visible_when_a_later_request_fails(capsys):
    result = {"status": "booked", "booking_id": "DEMO-0001", "flight_number": "SU123", "flight_date": "2027-06-10"}
    answer = {"diagnostic": {"source": "model", "category": "timeout", "action": "Проверьте подключение и повторите запрос."}}
    cli.print_result(answer, [{"tool": "book_flight", "executed": True, "result": result}])
    output = capsys.readouterr().out
    assert "Бронь создана" in output and "DEMO-0001" in output
    assert "Не удалось полностью обработать запрос" not in output and "Проверьте подключение" not in output


def test_unknown_booking_outcome_still_warns_before_retry():
    result = {
        "status": "error",
        "message": "Результат операции неизвестен. Проверьте состояние броней перед повторной отправкой.",
        "diagnostic": {"source": "tool", "category": "unexpected", "code": "runtime_error"},
    }
    trace = [{"tool": "book_flight", "executed": True, "result": result}]
    assert cli.format_result(result, trace) == result["message"]


def test_profile_results_through_cli(monkeypatch, capsys, tmp_path):
    from flight_agent.data import FlightStore

    store = FlightStore()
    operations = iter([(name, getattr(store, name)(*args)) for name, args in [
        ("set_profile", ("Анна",)), ("get_profile", ()), ("delete_profile", ()), ("get_profile", ())
    ]])
    def respond(*args):
        name, result = next(operations)
        return {"status": "ok"}, [{"tool": name, "result": result}]

    try:
        setup_cli(monkeypatch, ["имя", "имя", "удалить", "имя", "/exit"], respond, tmp_path)
        assert cli.main() == 0
        output = capsys.readouterr().out
        for expected in ("Имя сохранено: Анна.", "Ваше имя: Анна.", "Профиль удалён.", "Имя пользователя не сохранено."):
            assert expected in output
        assert "Не удалось получить сведения" not in output
    finally:
        store.close()
