from dataclasses import dataclass, field


@dataclass
class FilterInfo:
    methods: list[str] = field(default_factory=list)
    ingr_exclude: list[str] = field(default_factory=list)
    calories_range: tuple[int, int] = (0, 1000)
    proteins_range: tuple[int, int] = (0, 40)
    fats_range: tuple[int, int] = (0, 50)
    carbs_range: tuple[int, int] = (0, 100)
    time: float = 0
