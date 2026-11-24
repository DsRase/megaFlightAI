"""Безопасные сообщения об ошибках без текста исключений и данных запроса."""

import openai


def diagnostic(exc, source):
    status = getattr(exc, "status_code", None) if isinstance(exc, openai.APIStatusError) else None
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
        status = None
    if isinstance(exc, openai.APITimeoutError):
        category, action = "timeout", "Проверьте подключение и повторите запрос."
    elif isinstance(exc, openai.APIConnectionError):
        category, action = "connection", "Проверьте подключение и адрес сервиса."
    elif status in (401, 403):
        category, action = "authentication", "Проверьте ключ и права доступа к сервису."
    elif status == 429:
        category, action = "rate_limit", "Подождите и повторите запрос; проверьте лимиты сервиса."
    elif status is not None and status >= 500:
        category, action = "service", "Повторите запрос позже; проверьте состояние сервиса."
    elif status is not None:
        category, action = "request", "Проверьте настройки модели и параметры запроса."
    else:
        category, action = "unexpected", "Повторите запрос; если ошибка повторяется, обратитесь к разработчику."
    result = {"source": source, "category": category, "action": action}
    if category == "unexpected":
        # Только фиксированный список встроенных типов, без имён пользовательских классов.
        result["code"] = next((name for cls, name in (
            (ValueError, "value_error"), (TypeError, "type_error"),
            (KeyError, "key_error"), (RuntimeError, "runtime_error"),
            (OSError, "os_error"),
        ) if type(exc) is cls), "other_error")
    if status is not None:
        result["http_status"] = status
    return result


def error_answer(exc, source):
    info = diagnostic(exc, source)
    return {"status": "error", "message": "Не удалось обработать запрос. " + info["action"],
            "missing_fields": [], "alternatives": [], "diagnostic": info}
