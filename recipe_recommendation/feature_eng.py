import re
from datetime import datetime
from itertools import combinations
from typing import Final

import inflect
import pandas as pd
import torch
from pattern.text.en import singularize
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

COOKING_METH: Final = [
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


def clean_categories0(cat_str: str) -> str:
    lst = [el.strip() for el in cat_str.split(", ") if el.strip() != ""]
    lst = [el.replace(" Recipes", "") for el in lst if el != "Recipes"]
    cat_str = ", ".join(lst)
    return cat_str


def clean_categories1(cat_str: str) -> str:
    subcategories = re.split(r", | and ", cat_str)
    unique_subcategories = set(subcategories)
    cat_str = ", ".join(unique_subcategories)
    return cat_str


def generate_combinations(word1: str, word2: str, p: inflect.engine) -> list[str]:
    singular_word1 = singularize(word1) or word1
    singular_word2 = singularize(word2) or word2
    plural_word1 = p.plural(word1)
    plural_word2 = p.plural(word2)

    combinations_list = [
        f"{word1} {word2}",
        f"{word2} {word1}",
        f"{singular_word1} {singular_word2}",
        f"{word1} {singular_word2}",
        f"{singular_word1} {word2}",
        f"{singular_word2} {singular_word1}",
        f"{word2} {singular_word1}",
        f"{singular_word2} {word1}",
        f"{plural_word1} {plural_word2}",
        f"{word1} {plural_word2}",
        f"{plural_word1} {word2}",
        f"{plural_word2} {plural_word1}",
        f"{word2} {plural_word1}",
        f"{plural_word2} {word1}",
    ]
    return combinations_list


def singular_to_plural(cat_str: str) -> str:
    lst = cat_str.split(", ")
    p = inflect.engine()
    for i, el in enumerate(lst):
        words = el.split(" ")
        words[-1] = p.plural(words[-1])
        if len(words) > 1:
            lst[i] = " ".join(words)
    cat_str = ", ".join(set(lst))

    return cat_str


def clean_categories3(cat_str: str) -> str:
    lst = [cat.strip() for cat in cat_str.split(", ")]

    p = inflect.engine()
    found_combinations = set()

    for word1, word2 in combinations(lst, 2):
        try:
            combinations_list = generate_combinations(word1, word2, p)
            valid_combinations = [w for w in combinations_list if w in lst]
            found_combinations.update(valid_combinations)
        except IndexError:
            continue

    filtered_list = [x for x in lst if x not in found_combinations]

    return ", ".join(filtered_list)


def clean_categories4(cat_str: str) -> str:
    pattern = r"\b(?:" + "|".join(map(re.escape, CATEGORIES)) + r")\b"
    lst = cat_str.split(", ")
    cat_set = set()
    p = inflect.engine()
    for word in lst:
        if word in CATEGORIES or singularize(word) in CATEGORIES:
            cat_set.add(word)
            continue
        match = re.search(pattern, word)
        if match:
            parts = re.split(pattern, word, maxsplit=1)
            cat_set.add(parts[0].strip())
            matched_word = match.group()
            if matched_word not in cat_set or p.plural(matched_word) not in cat_set:
                cat_set.add(p.plural(matched_word))
        else:
            cat_set.add(word)

    cat_set.discard("")
    return ", ".join(cat_set)


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
        batch_results = CLASSIFIER(batch, COOKING_METH)
        results.extend([res["labels"][0] for res in batch_results])
    return results


def get_ingr_cat(ingredients: str) -> list[str]:
    ingr_lst = ingredients.split(",  ")
    result = CLASSIFIER(ingr_lst, INGREDIENTS)
    res_cat_ingr = [
        res["labels"][0] for res in result if res["scores"][0] > SCORE_INGR_CLASSIFIER_THRESHOLD
    ]
    return res_cat_ingr


def fill_cat_ingr(row: "pd.Series[float]", result_ingr: list[str]) -> "pd.Series[float]":
    row[result_ingr] = 1
    return row
