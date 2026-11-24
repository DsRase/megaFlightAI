import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, StrictStr

from . import BASE_URL, MODEL
from .__main__ import format_result
from .agent import MAX_MESSAGE_CHARS, TOOL_NAMES, create_agent, respond, safe_result
from .data import FlightStore
from .diagnostics import BOOKING_OUTCOME_UNKNOWN, diagnostic


STATIC = Path(__file__).resolve().parent / "static"
ALLOWED_HOSTS = {"localhost:8080", "127.0.0.1:8080"}
ALLOWED_ORIGINS = {"http://" + host for host in ALLOWED_HOSTS}
LOCK_TIMEOUT = 5
CHAT_LOG = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app):
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise RuntimeError("Установите переменную окружения LLM_API_KEY.")
    db_path = os.environ.get("FLIGHT_DB_PATH", "/data/flight.sqlite3")
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        store = FlightStore(db_path)
        try:
            model = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=key, temperature=None, timeout=60, max_retries=1)
            app.state.graph = create_agent(model, store)
        except Exception:
            store.close()
            raise
    except Exception:
        raise RuntimeError("Не удалось запустить приложение.") from None
    try:
        app.state.store = store
        app.state.lock = threading.Lock()
        app.state.thread_id = str(uuid.uuid4())
        yield
    finally:
        store.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def local_requests_only(request: Request, call_next):
    chat_request = request.method == "POST" and request.scope["path"] == "/api/chat"
    if chat_request:
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        CHAT_LOG.info("chat started request_id=%s", request_id)
    hosts = request.headers.getlist("host")
    origins = request.headers.getlist("origin")
    if len(hosts) != 1 or hosts[0].lower() not in ALLOWED_HOSTS or (
        origins and (len(origins) != 1 or origins[0].lower() not in ALLOWED_ORIGINS)
    ):
        response = JSONResponse({"reply": "Запрос отклонён."}, status_code=403)
    else:
        response = await call_next(request)
    if chat_request:
        status = getattr(request.state, "chat_status", "-")
        tool_statuses = getattr(request.state, "chat_tools", [])
        diagnostics = getattr(request.state, "chat_diagnostics", [])
        CHAT_LOG.log(
            logging.WARNING if diagnostics else logging.INFO,
            "chat completed request_id=%s response_http_status=%d result_status=%s elapsed_ms=%.0f tools=%s diagnostics=%s",
            request_id, response.status_code, status, (time.monotonic() - started) * 1000, tool_statuses, diagnostics,
        )
    return response


def record_chat_result(state, answer, trace):
    state.chat_tools = []
    state.chat_diagnostics = []
    records = [(None, answer)]
    records.extend((item.get("tool", "unknown"), item.get("result")) for item in (trace or []))
    for tool, result in records:
        if not isinstance(result, dict):
            continue
        checked = safe_result({key: result.get(key) for key in ("status", "diagnostic")})
        status = checked.get("status", "-")
        if tool is None:
            state.chat_status = status
        else:
            state.chat_tools.append(f"{tool if tool in TOOL_NAMES else 'unknown'}:{status}")
        info = checked.get("diagnostic", {})
        info = {key: info[key] for key in ("source", "category", "code", "http_status") if key in info}
        if info and info not in state.chat_diagnostics:
            state.chat_diagnostics.append(info)


class ChatRequest(BaseModel):
    message: StrictStr


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    return JSONResponse({"reply": "Ожидается JSON с текстовым полем message."}, status_code=422)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/app.js")
def script():
    return FileResponse(STATIC / "app.js", media_type="text/javascript")


@app.get("/styles.css")
def styles():
    return FileResponse(STATIC / "styles.css", media_type="text/css")


@app.post("/api/chat")
def chat(request: ChatRequest, http_request: Request):
    if not request.message.strip():
        return JSONResponse({"reply": "Введите сообщение."}, status_code=400)
    if len(request.message) > MAX_MESSAGE_CHARS:
        return JSONResponse({"reply": f"Сообщение должно содержать не более {MAX_MESSAGE_CHARS} символов."}, status_code=400)
    try:
        if not app.state.lock.acquire(timeout=LOCK_TIMEOUT):
            return JSONResponse({"reply": "Приложение занято. Попробуйте ещё раз."}, status_code=503)
        try:
            answer, trace = respond(app.state.graph, request.message, app.state.thread_id)
            record_chat_result(http_request.state, answer, trace)
            reply = format_result(answer, trace, app.state.store)
        finally:
            app.state.lock.release()
        return {"reply": reply}
    except Exception as exc:
        http_request.state.chat_status = "error"
        diagnostics = getattr(http_request.state, "chat_diagnostics", [])
        info = diagnostic(exc, "application")
        info.pop("action")
        http_request.state.chat_diagnostics = [*diagnostics, info]
        return JSONResponse({"reply": BOOKING_OUTCOME_UNKNOWN}, status_code=500)


@app.post("/api/chat/clear")
def clear_chat():
    try:
        if not app.state.lock.acquire(timeout=LOCK_TIMEOUT):
            return JSONResponse({"reply": "Приложение занято. Попробуйте ещё раз."}, status_code=503)
        try:
            app.state.graph.checkpointer.delete_thread(app.state.thread_id)
            app.state.thread_id = str(uuid.uuid4())
        finally:
            app.state.lock.release()
        return {"reply": "Диалог очищен. Профиль и брони сохранены."}
    except Exception:
        return JSONResponse({"reply": "Не удалось очистить диалог. Попробуйте ещё раз."}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080, access_log=False)
