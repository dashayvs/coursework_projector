import pandas as pd
from sklearn.preprocessing import LabelEncoder

from recipe_recommendation.feature_eng import (
    COURSES,
    MEALS,
    VEG_TYPES,
    clean_categories0,
    clean_categories1,
    clean_categories3,
    clean_categories4,
    convert_time_to_minutes,
    fill_cat_ingr,
    get_category,
    get_ingr_cat,
    get_type_cooking_batch,
    singular_to_plural,
)
from recipe_recommendation.paths import RAW_RECIPES_PATH

data = pd.read_csv(RAW_RECIPES_PATH)

data["Categories"] = (
    data["Categories"]
    .apply(clean_categories0)
    .apply(clean_categories1)
    .apply(singular_to_plural)
    .apply(clean_categories3)
    .apply(clean_categories4)
)

train_data = pd.DataFrame()
train_data[["Calories", "Fat", "Carbs", "Protein"]] = data.loc[
    :, ["Calories", "Fat", "Carbs", "Protein"]
]
train_data["Meal"] = data["Categories"].apply(get_category, args=(MEALS,))
train_data["Course"] = data["Categories"].apply(get_category, args=(COURSES,))
train_data["Healthy"] = data["Categories"].str.contains("Healthy").astype(int)
train_data["Special Nutrition"] = data["Categories"].apply(get_category, args=(VEG_TYPES,))
train_data["time, mins"] = data["Total Time"].apply(convert_time_to_minutes)
train_data[["Vegetables", "Fruits", "Meat", "Mushrooms", "Dairy", "Grains", "Nuts"]] = 0
train_data["Cooking Methods"] = get_type_cooking_batch(data["Directions"].tolist())

result_ingredients = list(data["Ingr"].apply(get_ingr_cat))
train_data = train_data.apply(lambda row: fill_cat_ingr(row, result_ingredients[row.name]), axis=1)

new_order = [
    "Calories",
    "Protein",
    "Fat",
    "Carbs",
    "Meal",
    "Course",
    "Healthy",
    "Special Nutrition",
    "Cooking Methods",
    "time, mins",
    "Vegetables",
    "Fruits",
    "Meat",
    "Seafood",
    "Mushrooms",
    "Dairy",
    "Grains",
    "Nuts",
]

train_data = train_data[new_order]
# train_data.to_csv("filter_data_recipes.csv", index=False)

le_enc = LabelEncoder()
train_data[["Meal", "Course", "Special Nutrition", "Cooking Methods"]] = train_data[
    ["Meal", "Course", "Special Nutrition", "Cooking Methods"]
].apply(lambda col: le_enc.fit_transform(col), axis=0)
# train_data.to_csv("train_data_recipes_encoded.csv", index=False)

data_text = data.loc[:, ["Directions", "Ingr", "URL"]]
# data_text.to_csv("train_data_text_url.csv", index=False)