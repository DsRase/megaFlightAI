import pytest


def pytest_configure(config):
    config.flight_metrics = []


@pytest.fixture
def metrics(request):
    return request.config.flight_metrics


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    rows = config.flight_metrics
    if not rows:
        return
    terminalreporter.section("Метрики сценариев с заранее заданной ScriptedModel (не качество выбора реальной модели)")
    for label, field in [("Правильный инструмент", "tool"), ("Правильные параметры", "args"), ("Корректный ответ", "answer")]:
        terminalreporter.write_line(f"{label}: {sum(bool(row[field]) for row in rows)}/{len(rows)}")
    security = [r for r in rows if r["security"]]
    terminalreporter.write_line(f"Безопасность: {sum(r['tool'] and r['args'] and r['answer'] for r in security)}/{len(security)}")
    terminalreporter.write_line("Общий результат всех тестов приведён в стандартной сводке pytest.")
