from __future__ import annotations

from typing import Optional


DEFAULT_MB_OUTPUT_FORMAT: str = "\t* Precision: {0} ({1} / {2})" \
                                "\n\t* Recall: {3} ({4} / {5})" \
                                "\n\t* F1: {6}"


class MatchBox:
    def __init__(self):
        super().__init__()
        self.score: int = 0
        self.hypothesis_count: int = 0
        self.reference_count: int = 0

    def __add__(self, other: MatchBox) -> MatchBox:
        new_match_box: MatchBox = MatchBox()
        if isinstance(other, MatchBox):
            new_match_box.score = self.score + other.score
            new_match_box.hypothesis_count = self.hypothesis_count + other.hypothesis_count
            new_match_box.reference_count = self.reference_count + other.reference_count
        else:
            raise TypeError(f"The value <{other}> is not a MatchBox. Please try again.")
        return new_match_box

    def __iadd__(self, other):
        if isinstance(other, MatchBox):
            self.score += other.score
            self.hypothesis_count += other.hypothesis_count
            self.reference_count += other.reference_count
        else:
            raise TypeError(f"The value <{other}> is not a MatchBox. Please try again.")
        return self

    def calculate_precision(self) -> float:
        if self.hypothesis_count < 0:
            raise ValueError("The sum of all hypotheses is negative, which should not be possible.")
        elif self.hypothesis_count == 0:
            precision = 0.0
        else:
            if self.score < 0:
                raise ValueError("The score is negative, which should not be possible.")
            else:
                precision: float = self.score / self.hypothesis_count

        return precision

    def calculate_recall(self) -> Optional[float]:
        if self.reference_count < 0:
            raise ValueError("The sum of all references is negative, which should not be possible.")
        elif self.reference_count == 0:
            recall: float = 0.0
        else:
            if self.score < 0:
                raise ValueError("The score is negative, which should not be possible.")
            else:
                recall: float = self.score / self.reference_count

        return recall

    def calculate_f_score(self, beta: float = 1) -> Optional[float]:
        precision: Optional[float] = self.calculate_precision()
        recall: Optional[float] = self.calculate_recall()

        if precision is None or recall is None:
            f_score: Optional[float] = None
        elif precision == 0.0 and recall == 0.0:
            f_score: float = 0.0
        else:
            f_score_numerator: float = precision * recall
            f_score_denominator: float = (beta**2 * precision) + recall
            f_score: float = (1 + beta**2) * (f_score_numerator / f_score_denominator)

        return f_score

    def calculate_statistics(self) -> list[float]:
        statistics: list[float] = [self.calculate_precision(), self.calculate_recall(), self.calculate_f_score()]
        return statistics

    def get_statistics_display(self, output_format: str = DEFAULT_MB_OUTPUT_FORMAT) -> str:
        stats_display_args: list[float] = [
            self.calculate_precision(), self.score, self.hypothesis_count,
            self.calculate_recall(), self.score, self.reference_count, self.calculate_f_score()
        ]
        stat_results: str = output_format.format(*stats_display_args)
        return stat_results
