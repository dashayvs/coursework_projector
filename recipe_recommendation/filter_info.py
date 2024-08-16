from dataclasses import dataclass


@dataclass
class FilterInfo:
    methods: list[str] | None = None
    ingr_exclude: list[str] | None = None
    calories_range: list[int] | None = None
    proteins_range: list[int] | None = None
    fats_range: list[int] | None = None
    carbs_range: list[int] | None = None
    time: int | None = None
