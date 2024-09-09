from __future__ import annotations

import pandas as pd
import pytest

from recipe_recommendation.feature_eng import (
    COURSES,
    MEALS,
    VEG_TYPES,
    # clean_categories0,
    # clean_categories1,
    # clean_categories3,
    convert_time_to_minutes,
    fill_cat_ingr,
    # generate_combinations,
    get_category,
)

# @pytest.mark.parametrize(
#     "raw_str, expected_categories",
#     [
#         (
#             "Dinner Recipes, Vegan, Vegetarian Recipes, Breakfast",
#             "Dinner, Vegan, Vegetarian, Breakfast",
#         ),
#         ("Recipes, Soup Recipes, Vegan", "Soup, Vegan"),
#         ("Lunch Recipes, Dinner Recipes, , Dessert", "Lunch, Dinner, Dessert"),
#     ],
# )
# def test_clean_categories0(raw_str, expected_categories):
#     assert clean_categories0(raw_str) == expected_categories


# @pytest.mark.parametrize(
#     "raw_str, expected_categories",
#     [
#         ("Dinner, Vegan, Dinner, Breakfast and Lunch", {"Breakfast", "Dinner", "Lunch", "Vegan"}),
#         ("Soup, Vegan, Soup, Vegan", {"Soup", "Vegan"}),
#     ],
# )
# def test_clean_categories1(raw_str, expected_categories) -> None:
#     assert set(clean_categories1(raw_str).split(", ")) == expected_categories


# @pytest.mark.parametrize(
#     "raw_str, expected_result",
#     [
#         ("Dinner, Vegan, Breakfast Recipe", {"Dinner", "Vegan", "Breakfast Recipes"}),
#         ("Soup Recipe, Vegan", {"Soup Recipes", "Vegan"}),
#     ],
# )
# def test_singular_to_plural(raw_str, expected_result) -> None:
#     assert set(singular_to_plural(raw_str).split(", ")) == expected_result


# def test_generate_combinations() -> None:
#     p = inflect.engine()
#     combinations_list = generate_combinations("Dinner", "Recipe", p)
#     assert "Dinner Recipe" in combinations_list
#     assert "Recipes Dinner" in combinations_list
#     assert "Dinners Recipes" in combinations_list


# @patch("recipe_recommendation.feature_eng.generate_combinations")
# def test_clean_categories3(mock_generate_combinations: Any) -> None:
#     mock_generate_combinations.side_effect = lambda word1, word2, p: [
#         f"{word1} {word2}",
#         f"{p.plural(word1)} {word2}",
#         f"{word1} {p.plural(word2)}",
#         f"{p.plural(word1)} {p.plural(word2)}",
#     ]
#
#     assert set(clean_categories3("Apple Pie, Apple, Pie, Dessert").split(", ")) == {
#         "Apple",
#         "Pie",
#         "Dessert",
#     }


@pytest.mark.parametrize(
    "raw_str, category, expected_result",
    [
        ("Dinner, Vegan", MEALS, "Dinner"),
        ("Lunch, Vegan", MEALS, "Lunch"),
        ("Soup, Vegan", MEALS, "None"),
        ("Main Dish, Vegan, Breakfast", COURSES, "Main Dish"),
        ("Side Dish, Vegan", COURSES, "Side Dish"),
        ("Apple, Vegan", COURSES, "None"),
        ("Dinner, Vegan, Breakfast", VEG_TYPES, "Vegan"),
        ("Lunch, Vegetarian", VEG_TYPES, "Vegetarian"),
        ("Soup, Meat", VEG_TYPES, "None"),
    ],
)
def test_get_category(raw_str, category, expected_result) -> None:
    assert get_category(raw_str, category) == expected_result


@pytest.mark.parametrize("time_str, expected_mins", [("1 hr 30 mins", 90), ("2 hrs 15 mins", 135)])
def test_convert_time_to_minutes(time_str, expected_mins) -> None:
    assert convert_time_to_minutes(time_str) == expected_mins


def test_fill_cat_ingr() -> None:
    data = pd.DataFrame({"name": ["test1", "test2"], "tomato": [0, 0], "potato": [0, 0]})
    result_ingr = [["tomato"], ["potato"]]
    data = data.apply(lambda row: fill_cat_ingr(row, result_ingr[row.name]), axis=1)
    assert data.loc[0, "tomato"] == 1
    assert data.loc[1, "potato"] == 1


if __name__ == "__main__":
    pytest.main()
