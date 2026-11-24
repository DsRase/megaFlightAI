import os
import threading
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
from .agent import MAX_MESSAGE_CHARS, create_agent, respond
from .data import FlightStore


STATIC = Path(__file__).resolve().parent / "static"
ALLOWED_HOSTS = {"localhost:8080", "127.0.0.1:8080"}
ALLOWED_ORIGINS = {"http://" + host for host in ALLOWED_HOSTS}
LOCK_TIMEOUT = 5


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
        app.state.lock = threading.Lock()
        app.state.thread_id = str(uuid.uuid4())
        yield
    finally:
        store.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def local_requests_only(request: Request, call_next):
    hosts = request.headers.getlist("host")
    origins = request.headers.getlist("origin")
    if len(hosts) != 1 or hosts[0].lower() not in ALLOWED_HOSTS or (
        origins and (len(origins) != 1 or origins[0].lower() not in ALLOWED_ORIGINS)
    ):
        return JSONResponse({"reply": "Запрос отклонён."}, status_code=403)
    return await call_next(request)


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
def chat(request: ChatRequest):
    if not request.message.strip():
        return JSONResponse({"reply": "Введите сообщение."}, status_code=400)
    if len(request.message) > MAX_MESSAGE_CHARS:
        return JSONResponse({"reply": f"Сообщение должно содержать не более {MAX_MESSAGE_CHARS} символов."}, status_code=400)
    try:
        if not app.state.lock.acquire(timeout=LOCK_TIMEOUT):
            return JSONResponse({"reply": "Приложение занято. Попробуйте ещё раз."}, status_code=503)
        try:
            answer, trace = respond(app.state.graph, request.message, app.state.thread_id)
            reply = format_result(answer, trace)
        finally:
            app.state.lock.release()
        return {"reply": reply}
    except Exception:
        return JSONResponse({"reply": "Не удалось получить результат. Проверьте состояние броней перед повторной отправкой."}, status_code=500)


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
