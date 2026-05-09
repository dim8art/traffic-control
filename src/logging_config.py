"""Единая настройка логирования: формат, уровень, stderr."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str = logging.INFO, *, force: bool = False) -> None:
    """
    Настройка корневого логгера (один раз, если уже вызывали — пропуск, кроме force=True).

    Точки входа приложения должны вызвать configure_logging до прочего кода;
    прикладные модули ограничиваются ``logging.getLogger(__name__)``.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if isinstance(level, int):
        lv = level
    else:
        key = str(level).upper()
        if sys.version_info >= (3, 11):
            # logging.getLevelNamesMapping() — официальный «сахар», без getattr(logging, "DEBUG").
            lv = logging.getLevelNamesMapping().get(key, logging.INFO)
        else:
            guess = getattr(logging, key, None)
            lv = guess if isinstance(guess, int) else logging.INFO
    kwargs: dict[str, object] = {
        "level": lv,
        "format": LOG_FORMAT,
        "datefmt": LOG_DATEFMT,
        "stream": sys.stderr,
    }
    if sys.version_info >= (3, 8):
        kwargs["force"] = True
    logging.basicConfig(**kwargs)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True
