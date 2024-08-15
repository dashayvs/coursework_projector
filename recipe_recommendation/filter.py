from os import PathLike
from pathlib import Path
from typing import AnyStr

import numpy as np
import numpy.typing as npt
import pandas as pd

from recipe_recommendation.filter_info import FilterInfo

ROOT_DIR = Path(__file__).parent.parent
F_DATA_PATH = ROOT_DIR / "data" / "filter_data_recipes.csv"


def filter_data(
    filter_info: FilterInfo,
    data_path: PathLike[AnyStr] = F_DATA_PATH,
) -> npt.NDArray[np.int64]:
    data = pd.read_csv(data_path)
    new_data = data.copy()

    if "Any" not in filter_info.methods and filter_info.methods is not None:
        new_data = new_data[new_data["Cooking Methods"].isin(filter_info.methods)]

    new_data = new_data[
        new_data[filter_info.ingr_exclude].apply(lambda row: all(val == 0 for val in row), axis=1)
    ]

    if filter_info.calories_range is not None and filter_info.calories_range != [0, 1000]:
        new_data = new_data.loc[
            (new_data["Calories"] >= filter_info.calories_range[0])
            & (new_data["Calories"] <= filter_info.calories_range[1])
        ]

    if filter_info.proteins_range is not None and filter_info.proteins_range != [0, 40]:
        new_data = new_data.loc[
            (new_data["Protein"] >= filter_info.proteins_range[0])
            & (new_data["Protein"] <= filter_info.proteins_range[1])
        ]

    if filter_info.fats_range is not None and filter_info.fats_range != [0, 50]:
        new_data = new_data.loc[
            (new_data["Fat"] >= filter_info.fats_range[0])
            & (new_data["Fat"] <= filter_info.fats_range[1])
        ]

    if filter_info.carbs_range is not None and filter_info.carbs_range != [0, 100]:
        new_data = new_data.loc[
            (new_data["Carbs"] >= filter_info.carbs_range[0])
            & (new_data["Carbs"] <= filter_info.carbs_range[1])
        ]

    if filter_info.time is not None and filter_info.time != 0:
        new_data = new_data.loc[new_data["time, mins"] <= filter_info.time]

    if new_data.shape[0] != data.shape[0]:
        return np.array(set(range(data.shape[0])) - set(list(new_data.index)), dtype=np.int64)
    else:
        return np.array([], dtype=np.int64)
