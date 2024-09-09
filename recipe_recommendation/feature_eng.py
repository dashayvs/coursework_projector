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


# def clean_categories0(cat_str: str) -> str:
#     # Split the input string cat_str by commas and remove any extra spaces around elements
#     lst = [el.strip() for el in cat_str.split(", ") if el.strip() != ""]
#
#     # Remove the word "Recipes" from each list element, and also filter out elements that are exactly "Recipes"
#     lst = [el.replace(" Recipes", "") for el in lst if el != "Recipes"]
#
#     cat_str = ", ".join(lst)
#     return cat_str

# def clean_categories1(cat_str: str) -> str:
#     # Split the string into subcategories by commas or the word "and".
#     subcategories = re.split(r", | and ", cat_str)
#     unique_subcategories = set(subcategories)
#     cat_str = ", ".join(unique_subcategories)
#     return cat_str

# def singular_to_plural(cat_str: str) -> str:
#     lst = cat_str.split(", ")
#     p = inflect.engine()
#
#     for i, el in enumerate(lst):
#         words = el.split(" ")
#
#         # Convert the last word to plural
#         words[-1] = p.plural(words[-1])
#
#         # If the category consists of more than one word, join the words back into a string
#         if len(words) > 1:
#             lst[i] = " ".join(words)
#
#     cat_str = ", ".join(set(lst))
#
#     return cat_str


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


# def generate_combinations(word1: str, word2: str, p: inflect.engine) -> list[str]:
#     singular_word1 = p.singular_noun(word1) or word1
#     singular_word2 = p.singular_noun(word2) or word2
#
#     plural_word1 = p.plural(word1)
#     plural_word2 = p.plural(word2)
#
#     combinations_list = [
#         f"{word1} {word2}",
#         f"{word2} {word1}",
#         f"{singular_word1} {singular_word2}",
#         f"{word1} {singular_word2}",
#         f"{singular_word1} {word2}",
#         f"{singular_word2} {singular_word1}",
#         f"{word2} {singular_word1}",
#         f"{singular_word2} {word1}",
#         f"{plural_word1} {plural_word2}",
#         f"{word1} {plural_word2}",
#         f"{plural_word1} {word2}",
#         f"{plural_word2} {plural_word1}",
#         f"{word2} {plural_word1}",
#         f"{plural_word2} {word1}",
#     ]
#     return combinations_list

# def clean_categories3(cat_str: str) -> str:
#     cat_str = " ".join(cat_str.split())
#     lst = [cat.strip() for cat in cat_str.split(",")]
#     p = inflect.engine()
#     found_combinations = set()
#
#     # Iterate over all unique pairs of categories in the list
#     for word1, word2 in combinations(lst, 2):
#         # Generate possible combinations for word1 and word2 (singular/plural, etc.)
#         combinations_list = generate_combinations(word1, word2, p)
#
#         # Filter the generated combinations to keep only those present in the output list
#         valid_combinations = [w for w in combinations_list if w in lst]
#         found_combinations.update(valid_combinations)
#
#     # Filter the original list to remove categories that are found in the set of combinations
#     filtered_list = [x for x in lst if x not in found_combinations]
#
#     return ", ".join(filtered_list)


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


# def clean_categories4(cat_str: str) -> str:
#     # Create a regex pattern to match any category in the CATEGORIES list
#     pattern = r"\b(?:" + "|".join(map(re.escape, CATEGORIES)) + r")\b"
#
#     cat_str = " ".join(cat_str.split())
#     lst = cat_str.split(",")
#     cat_set = set()
#     p = inflect.engine()
#
#     for word in lst:
#         # If the word is in CATEGORIES or its singular form is in CATEGORIES, add it to the set
#         if word in CATEGORIES or p.singular_noun(word) in CATEGORIES:
#             cat_set.add(word)
#             continue
#
#         # If the word matches the regex pattern, find the match
#         match = re.search(pattern, word)
#
#         if match:
#             # If a match is found, split the word by the regex pattern
#             parts = re.split(pattern, word, maxsplit=1)
#
#             # Add the first part to the set after stripping spaces
#             cat_set.add(parts[0].strip())
#             matched_word = match.group()
#
#             # If the matched word or its plural form are not already in the set, add the plural form
#             if matched_word not in cat_set or p.plural(matched_word) not in cat_set:
#                 cat_set.add(p.plural(matched_word))
#         else:
#             # If no match is found, add the word to the set
#             cat_set.add(word)
#
#     cat_set.discard("")
#
#     return ", ".join(cat_set)


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


def fill_cat_ingr(row: "pd.Series[float]", result_ingr: list[str]) -> "pd.Series[float]":
    row[result_ingr] = 1
    return row
