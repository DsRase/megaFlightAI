import builtins
import io
import os
import random
import re
import sys
import threading
import uuid
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from langchain_openai import ChatOpenAI
from . import BASE_URL, MODEL
from .agent import redact, respond, create_agent, safe_result, search_conditions, search_clarifications
from .diagnostics import error_answer
from .data import FlightStore

WAITING_MESSAGES = (
    "Сверяем курс", "Ищем попутный ветер", "Готовим взлёт", "Проверяем табло",
    "Считаем облака", "Прокладываем маршрут", "Расправляем крылья", "Ищем свободный гейт",
    "Прогреваем двигатели", "Ловим сигнал диспетчера", "Пристёгиваем мысли", "Заходим на посадку",
)


class PlainTextModel:
    """Добавить требование к оформлению, сохранив протокол агента."""

    def __init__(self, model):
        self.model = model

    def bind_tools(self, tools):
        return PlainTextModel(self.model.bind_tools(tools))

    def invoke(self, messages):
        messages = list(messages)
        messages[0] = messages[0].model_copy(update={"content": messages[0].content +
            "\nВ поле message отвечай обычным текстом, без Markdown и разметки."})
        return self.model.invoke(messages)


@contextmanager
def waiting_indicator():
    """Показывать только фиксированные сообщения, пока выполняется запрос."""
    stream = sys.stderr
    if not stream.isatty():
        yield
        return

    stopped = threading.Event()
    message = random.choice(WAITING_MESSAGES)
    frames = "/-\\|"
    width = len(message) + 2

    def animate():
        index = 0
        while not stopped.is_set():
            stream.write("\r" + f"{frames[index % len(frames)]} {message}".ljust(width))
            stream.flush()
            index += 1
            if stopped.wait(0.2):
                break

    worker = threading.Thread(target=animate, daemon=True)
    try:
        worker.start()
        yield
    finally:
        stopped.set()
        if worker.ident is not None:
            worker.join()
            stream.write("\r" + " " * width + "\r")
            stream.flush()


@contextmanager
def terminal_input():
    """Включить UTF-8 erase в терминале и восстановить исходные настройки."""
    try:
        import termios
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            raise OSError("not a tty")
        original = termios.tcgetattr(fd)
        updated = list(original)
        flag = getattr(termios, "IUTF8", 0x4000 if sys.platform.startswith("linux") else 0)
        if not flag:
            raise OSError("IUTF8 unavailable")
        updated[0] |= flag
        termios.tcsetattr(fd, termios.TCSANOW, updated)
    except (ImportError, OSError, ValueError, AttributeError):
        # Readline сам редактирует Unicode; если его нет, хотя бы не оставляем
        # кириллическое приглашение уязвимым к побайтовому стиранию.
        try:
            import readline  # noqa: F401
        except ImportError:
            pass
        yield
        return
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, original)


def print_result(answer, trace, stream=None):
    print = partial(builtins.print, file=stream)
    try:
        _print_result(answer, trace, stream)
    except Exception:
        booking = next((item.get("result", {}) for item in (trace or []) if isinstance(item, dict) and item.get("tool") == "book_flight" and item.get("executed") is not False and isinstance(item.get("result"), dict) and item["result"].get("status") in ("booked", "duplicate")), None)
        status = booking.get("status") if booking else None
        if status:
            print("Бронь создана." if status == "booked" else "Бронь для этого пассажира уже существует.")
            booking_id = booking.get("booking_id")
            if isinstance(booking_id, str) and re.fullmatch(r"DEMO-\d{4,}", booking_id):
                print("Номер брони:", booking_id)
        else:
            print("Не удалось обработать запрос. Попробуйте ещё раз.")


def plain_text(text):
    """Убрать распространённую разметку, не меняя одиночные символы внутри слов."""
    text = re.sub(r"(?m)^\s*```(?:[a-zA-Z0-9_-]+)?\s*$", "", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)
    text = re.sub(r"(?m)^\s{0,3}(?:-{3,}|\*{4,}|_{3,})\s*$", "", text)
    text = re.sub(r"(?m)^(.*)\n\s*(?:={3,}|-{3,})\s*$", r"\1", text)
    text = re.sub(r"(?m)^(\s*)[-*+]\s+", r"\1• ", text)
    text = re.sub(r"(?m)^(\s*• )\[[ xX]\]\s+", r"\1", text)
    text = re.sub(r"!\[([^\]\n]*)\]\((https?://[^\s)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)", r"\1 (\2)", text)
    # Маска чувствительных данных не является разметкой.
    chunks = text.split("***")
    for index, chunk in enumerate(chunks):
        chunk = re.sub(r"(?<!\w)(\*\*|__)(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", chunk)
        chunk = re.sub(r"~~(?=\S)(.+?)(?<=\S)~~", r"\1", chunk)
        chunks[index] = re.sub(r"(?<!\w)([*_`])(?=\S)([^\n]+?)(?<=\S)\1(?!\w)", r"\2", chunk)
    text = "***".join(chunks)
    return text.strip()


def print_flight(flight, prefix="", stream=None):
    print = partial(builtins.print, file=stream)
    parts = []
    for key, label in (("flight_number", "Рейс"), ("airline", "авиакомпания")):
        if flight.get(key):
            parts.append(f"{label} {flight[key]}")
    if flight.get("origin") or flight.get("destination"):
        parts.append(f"{flight.get('origin', '?')} — {flight.get('destination', '?')}")
    if flight.get("departure"):
        parts.append("вылет " + flight["departure"].replace("T", " "))
    if flight.get("status"):
        parts.append("статус: " + flight["status"])
    if "seats" in flight:
        parts.append(f"свободных мест: {flight['seats']}")
    if parts:
        print(prefix + "; ".join(parts))


def print_tool_result(name, result, stream=None):
    print = partial(builtins.print, file=stream)
    result = safe_result(result)
    message = result.get("message")
    if name in ("get_profile", "set_profile") and result.get("status") == "ok" and result.get("name"):
        print(("Ваше имя: " if name == "get_profile" else "Имя сохранено: ") + result["name"] + ".")
    elif name == "book_flight" and result.get("status") in ("booked", "duplicate"):
        print("Бронь создана." if result["status"] == "booked" else "Бронь для этого пассажира уже существует.")
        if result.get("booking_id") and result["booking_id"] != "***":
            print("Номер брони:", result["booking_id"])
        if result.get("flight_number"):
            print("Рейс:", result["flight_number"], result.get("flight_date", ""))
    elif name == "search_flights" and result.get("status") == "ok":
        flights = result.get("flights", [])
        if "count" in result:
            print("Нет подходящих рейсов." if result["count"] == 0 else f"Найдено рейсов: {result['count']}.")
        if result.get("truncated"):
            print(f"Показано: {len(flights)}.")
        for number, flight in enumerate(flights, 1):
            print_flight(flight, f"{number}. ", stream)
    elif name == "flight_status" and result.get("flight"):
        print_flight(result["flight"], stream=stream)
    elif name == "list_bookings" and result.get("status") == "ok":
        bookings = result.get("bookings", [])
        if not bookings:
            print("Активных броней нет.")
        for booking in bookings:
            print("Бронь:", booking.get("booking_id", "***"))
            print_flight({**booking, "status": booking.get("flight_status")}, stream=stream)
            if booking.get("flight_date") and not booking.get("departure"):
                print("Дата рейса:", booking["flight_date"])
    elif name == "cancel_booking" and result.get("status") == "cancelled":
        print("Бронь уже была отменена." if result.get("already_cancelled") else "Бронь отменена.")
        if result.get("booking_id") and result["booking_id"] != "***":
            print("Номер брони:", result["booking_id"])
    elif name == "update_booking" and result.get("status") == "updated":
        print("Рейс брони не изменился." if result.get("unchanged") else "Бронь перенесена.")
        if result.get("booking_id") and result["booking_id"] != "***":
            print("Номер брони:", result["booking_id"])
        if result.get("flight_number"):
            print("Рейс:", result["flight_number"], result.get("flight_date", ""))
    elif name == "update_booking" and result.get("status") == "duplicate":
        print("Перенос не выполнен: для этого пассажира уже есть бронь на новый рейс.")
    elif result.get("status") == "missing_fields" and message and message not in ("***", "Укажите недостающие данные."):
        print(plain_text(message))
    elif result.get("status") == "missing_fields" and result.get("missing_fields"):
        labels = {"flight_number": "номер рейса", "flight_date": "дату рейса", "passenger_name": "имя пассажира", "passport": "паспорт"}
        print("Укажите, пожалуйста: " + ", ".join(labels[field] for field in result["missing_fields"]) + ".")
    elif message and message != "***":
        print(plain_text(message))
    else:
        print("Не удалось получить сведения. Попробуйте ещё раз.")
    alternatives = result.get("alternatives", [])
    if alternatives:
        print(f"Подходящих альтернатив: {len(alternatives)}.")
        for number, flight in enumerate(alternatives, 1):
            print_flight(flight, f"{number}. ", stream)


def _print_result(answer, trace, stream=None):
    print = partial(builtins.print, file=stream)
    info = safe_result(answer).get("diagnostic", {})
    records = [item for item in (trace or []) if isinstance(item, dict)
               and isinstance(item.get("result"), dict)]
    if any(item.get("tool") == "book_flight" and item["result"].get("message") != "Допускается один вызов бронирования за ход." for item in records):
        records = [item for item in records if not (item.get("tool") == "book_flight" and item.get("executed") is False and item["result"].get("message") == "Допускается один вызов бронирования за ход.")]
    if any(item.get("tool") == "book_flight" and item["result"].get("status") in ("booked", "duplicate") for item in records):
        records = [item for item in records if item.get("tool") != "search_flights"]
    natural = answer.get("message", "")
    natural_tools = records and all(item.get("tool") in ("search_flights", "flight_status", "list_bookings") for item in records)
    def print_record(item):
        if item.get("tool") == "search_flights" and isinstance(item.get("arguments"), dict):
            print("Условия поиска: " + search_conditions(item["arguments"]) + ".")
        print_tool_result(item["tool"], item["result"], stream)

    if natural_tools and isinstance(natural, str) and natural and natural not in ("Поиск выполнен.", "Не удалось получить данные рейса.") and not info:
        if not all(item.get("tool") == "search_flights" for item in records) or "Ближайший рейс:" in natural:
            print(plain_text(redact(natural)))
        else:
            clarification = search_clarifications(natural)
            if clarification:
                print(plain_text(redact(clarification)))
        for item in records:
            args = item.get("arguments", {})
            if item.get("tool") == "search_flights" and args.get("limit") == 1 and args.get("from_now") is True and args.get("available_only") is True and item["result"].get("flights") and "Ближайший рейс:" in natural:
                continue
            print_record(item)
    elif records:
        for item in records:
            print_record(item)
    else:
        message = answer.get("message", "")
        if answer.get("status") in ("booked", "duplicate"):
            message = "Не удалось подтвердить результат бронирования. Попробуйте ещё раз."
        elif info.get("category") == "invalid_response":
            message = "Не удалось получить понятный ответ."
        print(plain_text(redact(message)) if isinstance(message, str) and message else "Не удалось получить ответ. Попробуйте ещё раз.")
    if info and records:
        print("Не удалось полностью обработать запрос.")
    action = info.get("action")
    if info.get("category") == "invalid_response":
        action = "Попробуйте ещё раз; если ошибка повторится, обратитесь к разработчику."
    if action and (records or action not in str(answer.get("message", ""))):
        print(plain_text(action))


def format_result(answer, trace):
    stream = io.StringIO()
    print_result(answer, trace, stream)
    return stream.getvalue().strip()


def main():
    key = os.environ.get("LLM_API_KEY")
    if not key:
        print("Установите переменную окружения LLM_API_KEY.")
        return 1
    try:
        model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=key, temperature=None, timeout=60, max_retries=1)
        db_path = os.environ.get("FLIGHT_DB_PATH", "./data/flights.sqlite3")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        store = FlightStore(db_path)
        graph = create_agent(PlainTextModel(model), store)
    except Exception as exc:
        print_result(error_answer(exc, "application"), [])
        return 1
    thread_id = str(uuid.uuid4())
    print("Учебный агент поможет узнать статус рейса и оформить демобронирование.")
    print("Рейсы и бронирования вымышленные; реальный билет не оформляется.")
    print("Используйте только вымышленные данные пассажира и паспорта (10 цифр).")
    print("Пример статуса: Какой статус рейса SU123 на 2027-06-10?")
    print("Пример бронирования: Забронируй SU124 на 2027-06-12.")
    print("Агент запросит недостающие данные. Выход: /exit, /quit, Ctrl-D или Ctrl-C.")
    with terminal_input():
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in {"/exit", "/quit"}:
                break
            if not text:
                continue
            try:
                with waiting_indicator():
                    answer, trace = respond(graph, text, thread_id)
            except KeyboardInterrupt:
                print()
                break
            except Exception as exc:
                print_result(error_answer(exc, "application"), None)
                continue
            print_result(answer, trace)
            if answer.get("_interrupted"):
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
