from dataclasses import dataclass


@dataclass
class FilterInfo:
    methods: list[str]
    ingr_exclude: list[str]
    calories_range: list[int]
    proteins_range: list[int]
    fats_range: list[int]
    carbs_range: list[int]
    time: int
