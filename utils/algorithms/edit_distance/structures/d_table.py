from typing import Sequence

from numpy import delete, int64
from numpy.typing import NDArray


class DTable:
    def __init__(self, chart: NDArray[int64]):
        self.chart = chart

    def __getitem__(self, indices: Sequence[int64]):
        return self.chart[indices]

    def __str__(self):
        rows, columns = self.chart.shape
        string_representation: str = f"D Table ({rows} x {columns}:\n" \
                                     f"{self.chart}"
        return string_representation

    def delete(self, deletion_slice: tuple[int, int, int], axis: int):
        self.chart = delete(self.chart, deletion_slice, axis)

    def repeat(self, repeat_indicator: NDArray[int], axis: int):
        self.chart = self.chart.repeat(repeat_indicator, axis)

    def shape(self):
        return self.chart.shape
