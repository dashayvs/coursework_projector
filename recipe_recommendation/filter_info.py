from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class Range:
    min: int
    max: int


# todo change default values
@dataclass
class FilterInfo:
    CALORIES_LIMITS: Final = Range(0, 1000)  # noqa: RUF009
    PROTEINS_LIMITS: Final = Range(0, 40)  # noqa: RUF009
    FATS_LIMITS: Final = Range(0, 50)  # noqa: RUF009
    CARBS_LIMITS: Final = Range(0, 100)  # noqa: RUF009

    methods: list[str] = field(default_factory=list)
    ingr_exclude: list[str] = field(default_factory=list)
    calories_range: Range = Range(0, 1000)  # noqa: RUF009
    proteins_range: Range = Range(0, 40)  # noqa: RUF009
    fats_range: Range = Range(0, 50)  # noqa: RUF009
    carbs_range: Range = Range(0, 100)  # noqa: RUF009
    time: float = 0
