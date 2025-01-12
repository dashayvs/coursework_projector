from os import PathLike

import numpy as np
import numpy.typing as npt
import pandas as pd

from recipe_recommendation.dataclasses import FilterInfo
from recipe_recommendation.paths import FILTER_DATA_PATH


def filter_data(
    filter_info: FilterInfo,
    data_path: PathLike[str] = FILTER_DATA_PATH,
) -> npt.NDArray[np.int64]:
    data = pd.read_csv(data_path)
    new_data = data.copy()

    if filter_info.methods != [] and "Any" not in filter_info.methods:
        new_data = new_data[new_data["Cooking Methods"].isin(filter_info.methods)]

    new_data = new_data[
        new_data[filter_info.ingr_exclude].apply(lambda row: all(val == 0 for val in row), axis=1)
    ]

    new_data = new_data.loc[
        (new_data["Calories"] >= filter_info.calories_range.min)
        & (new_data["Calories"] <= filter_info.calories_range.max)
    ]

    new_data = new_data.loc[
        (new_data["Protein"] >= filter_info.proteins_range.min)
        & (new_data["Protein"] <= filter_info.proteins_range.max)
    ]

    new_data = new_data.loc[
        (new_data["Fat"] >= filter_info.fats_range.min)
        & (new_data["Fat"] <= filter_info.fats_range.max)
    ]

    new_data = new_data.loc[
        (new_data["Carbs"] >= filter_info.carbs_range.min)
        & (new_data["Carbs"] <= filter_info.carbs_range.max)
    ]

    new_data = new_data.loc[new_data["time, mins"] <= filter_info.time]

    return new_data.index.to_numpy()
