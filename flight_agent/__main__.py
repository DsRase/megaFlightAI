import json
import os
import uuid

from langchain_openai import ChatOpenAI

from .agent import create_agent, respond


def main():
    key = os.environ.get("LLM_API_KEY")
    if not key:
        print("Установите переменную окружения LLM_API_KEY.")
        return 1
    model = ChatOpenAI(model="deepseek/deepseek-v4-flash", base_url="https://sprutdock.ru/v1", api_key=key, temperature=0, timeout=60, max_retries=1)
    graph = create_agent(model)
    thread_id = str(uuid.uuid4())
    print("Демо авиарейсов. Введите вопрос; выход: /exit или Ctrl-D.")
    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"/exit", "/quit"}:
            break
        if not text:
            continue
        try:
            answer, trace = respond(graph, text, thread_id)
        except Exception:
            # Исключения клиента могут содержать запрос, токены или личные данные.
            print("Ошибка обращения к агенту. Проверьте подключение и повторите запрос.")
            continue
        print("Инструменты:", json.dumps(trace, ensure_ascii=False))
        print("Ответ:", json.dumps(answer, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
