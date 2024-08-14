from recipe_recommendation.feature_eng import (
    clean_categories0,
    clean_categories1,
    singular_to_plural,
    clean_categories3,
    clean_categories4,
    get_meal,
    get_course,
    vegan_vegetarian,
    convert_time_to_minutes,
    get_type_cooking_batch,
    get_ingr_cat,
    fill_cat_ingr,
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv("../data_raw_recipes.csv")

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
train_data["Meal"] = data["Categories"].apply(get_meal)
train_data["Course"] = data["Categories"].apply(get_course)
train_data["Healthy"] = data["Categories"].str.contains("Healthy").astype(int)
train_data["Special Nutrition"] = data["Categories"].apply(vegan_vegetarian)
train_data["time, mins"] = data["Total Time"].apply(convert_time_to_minutes)
train_data[["Vegetables", "Fruits", "Meat", "Mushrooms", "Dairy", "Grains", "Nuts"]] = 0
train_data["Cooking Methods"] = get_type_cooking_batch(data["Directions"].tolist())

result_ingredients = list(data["Ingr"].apply(get_ingr_cat))
train_data = train_data.apply(lambda row: fill_cat_ingr(row, result_ingredients), axis=1)

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
