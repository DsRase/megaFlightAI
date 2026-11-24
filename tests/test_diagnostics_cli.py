import io
import json
import os
import sys

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

from flight_agent.__main__ import print_result, terminal_input
from flight_agent.agent import create_agent, respond
from flight_agent.diagnostics import diagnostic


class BrokenModel:
    def __init__(self, exc):
        self.exc = exc

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise self.exc


@pytest.mark.parametrize("status,category", [(401, "authentication"), (403, "authentication"), (429, "rate_limit"), (404, "request"), (503, "service")])
def test_model_http_errors_are_safe_and_informative(status, category):
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status, request=request)
    exc = openai.APIStatusError("sk-supersecret паспорт 1234567890", response=response, body=None)
    answer, trace = respond(create_agent(BrokenModel(exc)), "Вопрос")
    assert trace == []
    assert answer["diagnostic"]["category"] == category
    assert answer["diagnostic"]["http_status"] == status
    assert answer["diagnostic"]["action"] in answer["message"]
    assert "sk-supersecret" not in str(answer)
    assert "1234567890" not in str(answer)


def test_timeout_and_unexpected_errors_do_not_expose_exception():
    request = httpx.Request("POST", "https://example.invalid")
    assert diagnostic(openai.APITimeoutError(request=request), "model")["category"] == "timeout"
    answer, _ = respond(create_agent(BrokenModel(RuntimeError("sk-private 1234567890"))), "Вопрос")
    assert answer["diagnostic"]["category"] == "unexpected"
    assert "sk-private" not in str(answer)
    assert "1234567890" not in str(answer)


def test_model_error_after_successful_tool_keeps_result_and_diagnostic():
    class Model(BrokenModel):
        def invoke(self, messages):
            if messages[-1].type == "human":
                return AIMessage(content="", tool_calls=[{"name": "flight_status", "args": {"flight_number": "SU123", "flight_date": "2027-06-12"}, "id": "1", "type": "tool_call"}])
            raise self.exc

    answer, trace = respond(create_agent(Model(RuntimeError("secret"))), "Статус")
    assert answer["status"] == "ok"
    assert "SU123" in answer["message"]
    assert answer["diagnostic"]["source"] == "model"
    assert len(trace) == 1


def test_tool_error_safe_and_visible():
    class Store:
        def status(self, *args):
            raise RuntimeError("sk-secret паспорт 1234567890")

    class Model(BrokenModel):
        def invoke(self, messages):
            if messages[-1].type == "human":
                return AIMessage(content="", tool_calls=[{"name": "flight_status", "args": {"flight_number": "SU123"}, "id": "1", "type": "tool_call"}])
            return AIMessage(content=json.dumps({"status": "error"}))

    answer, trace = respond(create_agent(Model(None), Store()), "Статус")
    assert answer["diagnostic"]["source"] == "tool"
    assert trace[0]["result"]["diagnostic"]["category"] == "unexpected"
    assert "sk-secret" not in str([answer, trace])
    assert "1234567890" not in str([answer, trace])


def test_tool_details_are_not_displayed_and_answers_are_masked(capsys):
    print_result({"message": "Готово"}, [{"tool": "book_flight", "arguments": {"passport": "ABC", "api_key": "secret"}, "result": {"message": "паспорт 1234567890"}}])
    output = capsys.readouterr().out
    assert "book_flight" not in output
    assert "Параметры:" not in output and "Результат:" not in output
    assert "secret" not in output and "1234567890" not in output and "ABC" not in output
    print_result({"message": "Уточните вопрос"}, [])
    assert "Уточните вопрос" in capsys.readouterr().out
    print_result({"message": "Ошибка"}, None)
    assert "Ошибка" in capsys.readouterr().out


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux PTY IUTF8")
def test_terminal_sets_utf8_erase_and_restores_even_on_error(monkeypatch):
    import pty
    import termios

    master, slave = pty.openpty()
    stream = os.fdopen(os.dup(slave), "r")
    try:
        attrs = termios.tcgetattr(slave)
        attrs[0] &= ~getattr(termios, "IUTF8", 0x4000)
        attrs[3] |= termios.ICANON
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave, termios.TCSANOW, attrs)
        monkeypatch.setattr(sys, "stdin", stream)
        with pytest.raises(RuntimeError):
            with terminal_input():
                assert termios.tcgetattr(slave)[0] & getattr(termios, "IUTF8", 0x4000)
                erase = attrs[6][termios.VERASE]
                if isinstance(erase, int):
                    erase = bytes([erase])
                os.write(master, "аб".encode() + erase + b"\n")
                assert os.read(slave, 100) == "а\n".encode()
                raise RuntimeError()
        assert termios.tcgetattr(slave) == attrs
    finally:
        stream.close()
        os.close(master)
        os.close(slave)


def test_non_tty_input_is_untouched(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("вопрос"))
    with terminal_input():
        assert sys.stdin.read() == "вопрос"


def test_cli_whitelists_unknown_fields_names_and_arbitrary_strings(capsys):
    print_result({"message": "AB1234567890", "AB1234567890": "secret"}, [{
        "tool": "AB1234567890", "arguments": {"passenger_name": "AB1234567890", "AB1234567890": "secret", "flight_number": "AB1234567890"},
        "result": {"message": "AB1234567890", "AB1234567890": "secret"},
    }])
    output = capsys.readouterr().out
    assert "AB1234567890" not in output
    assert "secret" not in output
    assert "unknown" not in output


def test_common_exception_codes_never_include_exception_text():
    for exc, code in [(ValueError("private"), "value_error"), (TypeError("private"), "type_error"), (RuntimeError("private"), "runtime_error")]:
        info = diagnostic(exc, "application")
        assert info["code"] == code
        assert "private" not in str(info)


def test_cli_shows_natural_answers_and_diagnostic_action(capsys):
    action = "Проверьте подключение и повторите запрос."
    answer = {"status": "missing_fields", "message": "Введите паспорт повторно, пожалуйста.", "missing_fields": ["passport"], "diagnostic": {"source": "model", "category": "timeout", "action": action}}
    print_result(answer, [])
    output = capsys.readouterr().out
    assert "Введите паспорт повторно, пожалуйста." in output
    assert action in output
    print_result({"status": "ok", "message": "Рейс задержан; паспорт 1234567890", "flight": {"flight_number": "SU123"}}, [{"tool": "flight_status", "arguments": {}, "result": {"status": "ok", "flight": {"flight_number": "SU123"}}}])
    output = capsys.readouterr().out
    assert "SU123" in output and "1234567890" not in output
