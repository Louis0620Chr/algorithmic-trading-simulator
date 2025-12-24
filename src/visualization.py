import matplotlib.pyplot as plt
import numpy


def _format_percentage(value: float) -> str:
    if value is None or numpy.isnan(value):
        return "n/a"
    return "{:.2f}%".format(value * 100.0)


def _format_number(value: float) -> str:
    if value is None or numpy.isnan(value):
        return "n/a"
    return "{:.2f}".format(value)
