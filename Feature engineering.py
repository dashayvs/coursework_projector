import pandas as pd
import inflect
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
import torch
from itertools import combinations
from pattern.text.en import singularize
import re

data = pd.read_csv("../data_raw_recipes.csv")


def clean_categories0(cat_str):
    lst = [el.strip() for el in cat_str.split(", ") if el.strip() != ""]
    lst = [el.replace(" Recipes", "") for el in lst if el != "Recipes"]
    cat_str = ", ".join(lst)

    return cat_str


def clean_categories1(cat_str):
    subcategories = re.split(r', | and ', cat_str)
    unique_subcategories = set(subcategories)
    cat_str = ", ".join(unique_subcategories)

    return cat_str


def clean_categories2(cat_str):
    lst = cat_str.split(", ")

    unique_subcategories = set()

    for el in lst:
        el = el.strip()

        if ' ' in el:
            temp = el.split(' ')
            if not all(item in lst for item in temp):
                unique_subcategories.add(el)
        else:
            unique_subcategories.add(el)

    return ", ".join(unique_subcategories)


def singular_to_plural(cat_str):
    try:
        lst = cat_str.split(", ")
        p = inflect.engine()
        for i, el in enumerate(lst):
            words = el.split(" ")
            words[-1] = p.plural(words[-1])
            if " ".join(words) in lst:
                lst[i] = " ".join(words)

        cat_str = ", ".join(set(lst))
    except:
        print(cat_str)

    return cat_str


def generate_combinations(word1, word2, p):
    singular_word1 = singularize(word1)
    singular_word2 = singularize(word2)
    plural_word1 = p.plural(word1)
    plural_word2 = p.plural(word2)

    combinations_list = [
        f"{word1} {word2}", f"{word2} {word1}",
        f"{singular_word1} {singular_word2}", f"{word1} {singular_word2}",
        f"{singular_word1} {word2}", f"{singular_word2} {singular_word1}",
        f"{word2} {singular_word1}", f"{singular_word2} {word1}",
        f"{plural_word1} {plural_word2}", f"{word1} {plural_word2}",
        f"{plural_word1} {word2}", f"{plural_word2} {plural_word1}",
        f"{word2} {plural_word1}", f"{plural_word2} {word1}"
    ]

    return combinations_list


def clean_categories3(cat_str):
    try:
        lst = cat_str.split(", ")
        p = inflect.engine()
        found_combinations = []

        for word1, word2 in combinations(lst, 2):
            combinations_list = generate_combinations(word1, word2, p)
            found_combinations.extend(w for w in combinations_list if w in lst)

        lst = list(filter(lambda x: x not in found_combinations, lst))

        return ', '.join(lst)
    except:
        return cat_str


def clean_categories4(cat_str):
    lst_cat = ["Main Dish", "Appetizer", "Salad", "Dinner", "BBQ & Grilled", "Dessert",
               "Dressing", "Sandwich", "Side Dish", "Soup", "Lunch", "Breakfast"]

    pattern = r'\b(?:' + '|'.join(map(re.escape, lst_cat)) + r')\b'
    lst = cat_str.split(", ")
    cat_set = set()
    p = inflect.engine()
    for word in lst:
        if word in lst_cat or singularize(word) in lst_cat:
            cat_set.add(word)
            continue
        match = re.search(pattern, word)
        if match:
            parts = re.split(pattern, word, 1)
            cat_set.add(parts[0].strip())
            matched_word = match.group()
            if matched_word not in cat_set or p.plural(matched_word) not in cat_set:
                cat_set.add(p.plural(matched_word))
        else:
            cat_set.add(word)

    cat_set.discard("")
    return ", ".join(cat_set)


def get_meal(cat_str):
    meals = ["Breakfast", "Snack", "Lunch", "Brunch", "Dinner", "Supper"]
    for m in meals:
        if m in cat_str:
            return m
    return "None"


def get_course(cat_str):
    courses = ["Dessert", "Side Dish", "Salad", "Soup", "Main Dish", "Appetizer"]
    for m in courses:
        if m in cat_str:
            return m
    return "None"


def vegan_vegetarian(cat_str):
    lst = ['Vegan', 'Vegetarian']
    for m in lst:
        if m in cat_str:
            return m
    return "None"


def convert_time_to_minutes(time_str):
    parts = time_str.split()

    hours = 0
    minutes = 0

    for i in range(len(parts)):
        if parts[i] == "hrs":
            hours = int(parts[i - 1])
        elif parts[i] == "mins":
            minutes = int(parts[i - 1])

    total_minutes = hours * 60 + minutes

    return total_minutes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

candidate_labels_cook_meth = ['Baking', 'Boiling', 'Frying', 'Grilling', 'Steaming', 'Stewing',
                              'Raw Food', 'Sous Vide', 'Freezing', 'Smoking', 'Microwave']

candidate_labels_ingr = ["Vegetables", "Fruits", "Meat", "Seafood", "Mushrooms", "Dairy", "Grains", "Nuts"]


def get_type_cooking_batch(dir_str_list):
    batch_size = 32
    results = []

    for i in range(0, len(dir_str_list), batch_size):
        batch = dir_str_list[i:i + batch_size]
        batch_results = classifier(batch, candidate_labels_cook_meth)
        results.extend([res["labels"][0] for res in batch_results])

    return results


def get_ingr_cat(ingredients):
    res_cat_ingr = []
    ingr_lst = ingredients.split(",  ")

    result = classifier(ingr_lst, candidate_labels_ingr)
    for res in result:
        if res["scores"][0] > 0.6:
            res_cat_ingr.append(res["labels"][0])

    return set(res_cat_ingr)


result_cat_ingr = list(data["Ingr"].apply(get_ingr_cat))


def fill_cat_ingr(row):
    index = row.name

    for cat in result_cat_ingr[index]:
        row[cat] = int(1)

    return row


data["Categories"] = data["Categories"].apply(clean_categories0).apply(clean_categories1).apply(
    clean_categories2).apply(singular_to_plural).apply(clean_categories3).apply(clean_categories4)


train_data = pd.DataFrame()
train_data[['Calories', 'Fat', 'Carbs', 'Protein']] = data.loc[:, ['Calories', 'Fat', 'Carbs', 'Protein']]
train_data["Meal"] = data["Categories"].apply(get_meal)
train_data["Course"] = data["Categories"].apply(get_course)
train_data["Healthy"] = data["Categories"].str.contains("Healthy").astype(int)
train_data["Special Nutrition"] = data["Categories"].apply(vegan_vegetarian)
train_data["time, mins"] = data["Total Time"].apply(convert_time_to_minutes)
train_data[["Vegetables", "Fruits", "Meat", "Mushrooms", "Dairy", "Grains", "Nuts"]] = 0
train_data['Cooking Methods'] = get_type_cooking_batch(data["Directions"].tolist())

train_data = train_data.apply(fill_cat_ingr, axis=1)

new_order = ["Calories", "Protein", "Fat", "Carbs", "Meal", "Course", "Healthy", "Special Nutrition",
             "Cooking Methods", "time, mins", "Vegetables", "Fruits", "Meat", "Seafood", "Mushrooms", "Dairy", "Grains",
             "Nuts"]

train_data = train_data[new_order]

# train_data.to_csv("filter_data_recipes.csv", index=False)

le_enc = LabelEncoder()
train_data[["Meal","Course","Special Nutrition", "Cooking Methods"]] = train_data[["Meal","Course","Special Nutrition", "Cooking Methods"]].apply(lambda col: le_enc.fit_transform(col), axis=0)
# train_data.to_csv("train_data_recipes_encoded.csv", index=False)

data_text = data.loc[:,["Directions", "Ingr", "URL"]]
# data_text.to_csv("train_data_text_url.csv", index=False)
