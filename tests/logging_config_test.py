from __future__ import annotations

import logging

from src.logging_config import configure_logging


def test_configure_logging_sets_level_from_int() -> None:
    """При передаче уровня числом корень логгера получает тот же уровень."""
    configure_logging(logging.WARNING, force=True)
    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_sets_level_from_string_debug() -> None:
    """Строковое имя уровня (DEBUG) переводится в числовое значение логирования."""
    configure_logging("DEBUG", force=True)
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_unknown_string_name_defaults_to_info() -> None:
    """Неизвестное имя уровня не ломает настройку: остаётся безопасный INFO."""
    configure_logging("NO_SUCH_LEVEL_XYZ", force=True)
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_twice_without_force_does_not_add_extra_handlers() -> None:
    """Второй configure_logging(force=False) ничего не делает и не добавляет второй stderr-хендлер."""
    configure_logging(force=True)
    handlers_after_first = len(logging.getLogger().handlers)
    configure_logging(force=False)
    assert len(logging.getLogger().handlers) == handlers_after_first
