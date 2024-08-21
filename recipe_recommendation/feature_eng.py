import re
from datetime import datetime
from itertools import combinations
from typing import cast

import inflect
import pandas as pd
import torch
from pattern.text.en import singularize
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

COOK_METH = [
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

INGREDIENTS = [
    "Vegetables",
    "Fruits",
    "Meat",
    "Seafood",
    "Mushrooms",
    "Dairy",
    "Grains",
    "Nuts",
]

CATEGORIES = [
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

MEALS = ["Breakfast", "Snack", "Lunch", "Brunch", "Dinner", "Supper"]

COURSES = ["Dessert", "Side Dish", "Salad", "Soup", "Main Dish", "Appetizer"]

SCORE_INGR_CLASSIFIER_THRESHOLD = 0.6


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
    singular_word1 = singularize(word1)
    singular_word2 = singularize(word2)
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
    try:
        lst = cat_str.split(", ")
        p = inflect.engine()
        for i, el in enumerate(lst):
            words = el.split(" ")
            words[-1] = p.plural(words[-1])
            if len(words) > 1:
                lst[i] = " ".join(words)
        cat_str = ", ".join(set(lst))
    except Exception as e:
        print(e)
        print(cat_str)
    return cat_str


def clean_categories3(cat_str: str) -> str:
    try:
        lst = cat_str.split(", ")
        p = inflect.engine()
        found_combinations: list[str] = []

        for word1, word2 in combinations(lst, 2):
            combinations_list: list[str] = generate_combinations(word1, word2, p)
            found_combinations.extend(w for w in combinations_list if w in lst)

        lst = list(filter(lambda x: x not in found_combinations, lst))
        return ", ".join(lst)
    except Exception as e:
        print(e)
        return cat_str


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


def get_meal(cat_str: str) -> str:
    for m in MEALS:
        if m in cat_str:
            return m
    return "None"


def get_course(cat_str: str) -> str:
    for m in COURSES:
        if m in cat_str:
            return m
    return "None"


def vegan_vegetarian(cat_str: str) -> str:
    for m in ["Vegan", "Vegetarian"]:
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
        batch_results = CLASSIFIER(batch, COOK_METH)
        results.extend([res["labels"][0] for res in batch_results])
    return results


def get_ingr_cat(ingredients: str) -> list[str]:
    ingr_lst = ingredients.split(",  ")
    result = CLASSIFIER(ingr_lst, INGREDIENTS)
    res_cat_ingr = [
        res["labels"][0] for res in result if res["scores"][0] > SCORE_INGR_CLASSIFIER_THRESHOLD
    ]
    return res_cat_ingr


def fill_cat_ingr(row: pd.Series[float], result_ingr: list[list[str]]) -> pd.Series[float]:
    index = cast(int, row.name)
    for cat in result_ingr[index]:
        row[cat] = 1
    return row
