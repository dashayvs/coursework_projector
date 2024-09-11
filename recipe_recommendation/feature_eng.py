import re
from datetime import datetime
from itertools import combinations
from typing import Final

import inflect
import pandas as pd
import torch
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

COOKING_METHODS: Final = [
    "Baking",
    "Boiling",
    "Frying",
    "Grilling",
    "Steaming",
    "Stewing",
    "Blending",
    "Raw Food",
    "Sous Vide",
    "Freezing",
    "Smoking",
    "Microwave",
]

INGREDIENTS: Final = [
    "Vegetables",
    "Fruits",
    "Meat",
    "Seafood",
    "Mushrooms",
    "Dairy",
    "Grains",
    "Nuts",
]

CATEGORIES: Final = [
    "Main Dish",
    "Appetizer",
    "Salad",
    "Dinner",
    "BBQ & Grilled",
    "Dessert",
    "Dressing",
    "Sandwich",
    "Side Dish",
    "Soup",
    "Lunch",
    "Breakfast",
]

MEALS: Final = ["Breakfast", "Snack", "Lunch", "Brunch", "Dinner", "Supper"]

COURSES: Final = ["Dessert", "Side Dish", "Salad", "Soup", "Main Dish", "Appetizer"]

VEG_TYPES: Final = ["Vegan", "Vegetarian"]

SCORE_INGR_CLASSIFIER_THRESHOLD: Final = 0.6


def normalize_categories(cat_str: str) -> set[str]:
    p = inflect.engine()

    # Normalize spaces and split by commas or 'and'
    categories = re.split(r"\s*,\s*|\s+and\s+", " ".join(cat_str.split()))

    # Remove 'Recipes', strip whitespace, and apply singularization
    new_categories = set()
    for category in categories:
        new_category = category.replace("Recipes", "").strip()
        if new_category:  # Ensure the category is not an empty string
            singularized = " ".join(
                [p.singular_noun(word) or word for word in new_category.split()]
            )
            new_categories.add(singularized)

    return new_categories


def filter_out_combined_categories(categories: set[str]) -> set[str]:
    found_combinations = set()

    # Generate and check combinations
    for word1, word2 in combinations(categories, 2):
        combined1 = f"{word1} {word2}"
        combined2 = f"{word2} {word1}"

        if combined1 in categories:
            found_combinations.add(combined1)
        if combined2 in categories:
            found_combinations.add(combined2)

    return categories - found_combinations


def split_elements_by_categories(elements: set[str]) -> str:
    split_elements = set()

    pattern = r"\b(?:" + "|".join(map(re.escape, CATEGORIES)) + r")\b"

    for element in elements:
        # Find all matches for the CATEGORIES in the element
        matches = re.findall(pattern, element)
        if matches:
            # Add the matched CATEGORIES to the result set
            split_elements.update(matches)

            # Split the element by all matched CATEGORIES
            parts = re.split(pattern, element)

            # Add non-empty parts to the result set
            split_elements.update(part.strip() for part in parts if part.strip())
        else:
            # If no category is found, just add the element as is
            split_elements.add(element)

    return ", ".join(split_elements)


def preprocess_categories(cat_str: str) -> str:
    return split_elements_by_categories(
        filter_out_combined_categories(normalize_categories(cat_str))
    )


def get_category(cat_str: str, cat_name: list[str]) -> str:
    for m in cat_name:
        if m in cat_str:
            return m
    return "None"


def convert_time_to_minutes(time_str: str) -> int:
    formatted_str = (
        time_str.replace(" hrs", "H").replace(" hr", "H").replace(" mins", "M").replace(" min", "M")
    )
    dt: datetime = datetime.strptime(formatted_str, "%HH %MM")
    return dt.hour * 60 + dt.minute


def get_type_cooking_batch(dir_str_list: list[str]) -> list[str]:
    batch_size = 32
    results = []
    for i in range(0, len(dir_str_list), batch_size):
        batch = dir_str_list[i : i + batch_size]
        batch_results = CLASSIFIER(batch, COOKING_METHODS)
        results.extend([res["labels"][0] for res in batch_results])
    return results


def get_ingr_cat(ingredients: str) -> list[str]:
    ingr_lst = ingredients.split(",  ")
    result = CLASSIFIER(ingr_lst, INGREDIENTS)
    res_cat_ingr = [
        res["labels"][0] for res in result if res["scores"][0] > SCORE_INGR_CLASSIFIER_THRESHOLD
    ]
    return res_cat_ingr


def fill_cat_ingr(row: "pd.Series[float]", recipe_ingr: list[str]) -> "pd.Series[float]":
    row[recipe_ingr] = 1
    return row
