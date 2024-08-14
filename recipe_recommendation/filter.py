from typing import List, Any
import numpy.typing as npt
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def filter_data(
    param_list: List[Any],
    data_path: str = "D:\\recipe_recomendation\\data\\filter_data_recipes.csv",
) -> npt.NDArray[np.int64]:
    data = pd.read_csv(data_path)
    new_data = data.copy()

    if "Any" not in param_list[0] and param_list[0] != []:
        new_data = new_data[new_data["Cooking Methods"].isin(param_list[0])]

    new_data = new_data[
        new_data[param_list[1]].apply(lambda row: all(val == 0 for val in row), axis=1)
    ]

    if param_list[2] != [] and param_list[2] != [0, 1000]:
        new_data = new_data.loc[
            (new_data["Calories"] >= param_list[2][0]) & (new_data["Calories"] <= param_list[2][1])
        ]

    if param_list[3] != [] and param_list[3] != [0, 40]:
        new_data = new_data.loc[
            (new_data["Protein"] >= param_list[3][0]) & (new_data["Protein"] <= param_list[3][1])
        ]

    if param_list[4] != [] and param_list[4] != [0, 50]:
        new_data = new_data.loc[
            (new_data["Fat"] >= param_list[4][0]) & (new_data["Fat"] <= param_list[4][1])
        ]

    if param_list[5] != [] and param_list[5] != [0, 100]:
        new_data = new_data.loc[
            (new_data["Carbs"] >= param_list[5][0]) & (new_data["Carbs"] <= param_list[5][1])
        ]

    if param_list[6] != [] and param_list[6] != [0]:
        new_data = new_data.loc[new_data["time, mins"] <= param_list[6][0]]

    if new_data.shape[0] != data.shape[0]:
        return np.array(set(range(data.shape[0])) - set(list(new_data.index)), dtype=np.int64)
    else:
        return np.array([], dtype=np.int64)
