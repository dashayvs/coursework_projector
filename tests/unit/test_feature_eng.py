from typing import Any
from unittest.mock import patch

import inflect
import pandas as pd
import pytest

# Assuming these functions are defined in recipe_recommendation.feature_eng_funcs
from recipe_recommendation.feature_eng import (
    clean_categories0,
    clean_categories1,
    clean_categories3,
    convert_time_to_minutes,
    fill_cat_ingr,
    generate_combinations,
    get_course,
    get_meal,
    singular_to_plural,
    vegan_vegetarian,
)


def test_clean_categories0():
    assert (
        clean_categories0("Dinner Recipes, Vegan, Vegetarian Recipes, Breakfast")
        == "Dinner, Vegan, Vegetarian, Breakfast"
    )
    assert clean_categories0("Recipes, Soup Recipes, Vegan") == "Soup, Vegan"
    assert clean_categories0("Lunch Recipes, Dinner Recipes, , Dessert") == "Lunch, Dinner, Dessert"


def test_clean_categories1() -> None:
    assert set(clean_categories1("Dinner, Vegan, Dinner, Breakfast and Lunch").split(", ")) == {
        "Breakfast",
        "Dinner",
        "Lunch",
        "Vegan",
    }
    assert set(clean_categories1("Soup, Vegan, Soup, Vegan").split(", ")) == {
        "Soup",
        "Vegan",
    }


def test_singular_to_plural() -> None:
    assert set(singular_to_plural("Dinner, Vegan, Breakfast Recipe").split(", ")) == {
        "Dinner",
        "Vegan",
        "Breakfast Recipes",
    }
    assert set(singular_to_plural("Soup Recipe, Vegan").split(", ")) == {
        "Soup Recipes",
        "Vegan",
    }


def test_generate_combinations() -> None:
    p = inflect.engine()
    combinations_list = generate_combinations("Dinner", "Recipe", p)
    assert "Dinner Recipe" in combinations_list
    assert "Recipes Dinner" in combinations_list
    assert "Dinners Recipes" in combinations_list


@patch("recipe_recommendation.feature_eng.generate_combinations")
def test_clean_categories3(mock_generate_combinations: Any) -> None:
    mock_generate_combinations.side_effect = lambda word1, word2, p: [
        f"{word1} {word2}",
        f"{p.plural(word1)} {word2}",
        f"{word1} {p.plural(word2)}",
        f"{p.plural(word1)} {p.plural(word2)}",
    ]

    assert set(clean_categories3("Apple Pie, Apple, Pie, Dessert").split(", ")) == {
        "Apple",
        "Pie",
        "Dessert",
    }


def test_get_meal() -> None:
    assert get_meal("Dinner, Vegan") == "Dinner"
    assert get_meal("Lunch, Vegan") == "Lunch"
    assert get_meal("Soup, Vegan") == "None"


def test_get_course() -> None:
    assert get_course("Main Dish, Vegan, Breakfast") == "Main Dish"
    assert get_course("Side Dish, Vegan") == "Side Dish"
    assert get_course("Apple, Vegan") == "None"


def test_vegan_vegetarian() -> None:
    assert vegan_vegetarian("Dinner, Vegan, Breakfast") == "Vegan"
    assert vegan_vegetarian("Lunch, Vegetarian") == "Vegetarian"
    assert vegan_vegetarian("Soup, Meat") == "None"


def test_convert_time_to_minutes() -> None:
    assert convert_time_to_minutes("1 hr 30 mins") == 90
    assert convert_time_to_minutes("2 hrs 15 mins") == 135


def test_fill_cat_ingr() -> None:
    data = pd.DataFrame({"name": ["test1", "test2"], "tomato": [0, 0], "potato": [0, 0]})
    result_ingr = [["tomato"], ["potato"]]
    data = data.apply(lambda row: fill_cat_ingr(row, result_ingr), axis=1)
    assert data.loc[0, "tomato"] == 1
    assert data.loc[1, "potato"] == 1


if __name__ == "__main__":
    pytest.main()
