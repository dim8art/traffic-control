"""Обратная совместимость: MaxPressure перенесён в phase_control."""

from src.simulation.phase_control import MaxPressureController

__all__ = ["MaxPressureController"]
