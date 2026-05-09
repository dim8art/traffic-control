"""Изоляция глобального состояния логирования между тестами."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _reset_logging_state() -> None:
    """Сбрасывает состояние root-логгера и флаг configure_logging между тестами.

    Чтобы последовательные вызовы basicConfig/хендлеры не портили следующий кейс.
    """
    import src.logging_config as lc

    lc._CONFIGURED = False
    root = logging.getLogger()
    root.handlers[:] = []
    yield
    root.handlers[:] = []
    lc._CONFIGURED = False
