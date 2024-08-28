from dataclasses import astuple
from typing import Final

import pandas as pd
import streamlit as st

from recipe_recommendation.filter import filter_data
from recipe_recommendation.filter_info import FilterInfo, Range
from recipe_recommendation.models import ObjectsTextSimilarity
from recipe_recommendation.paths import MODEL_PATH, RECIPES_PATH
from recipe_recommendation.recipe_info import RecipeInfo

N_FILTERED_RECIPES_THRESHOLD: Final = 100

if "model" not in st.session_state:
    st.session_state.model = ObjectsTextSimilarity.load(MODEL_PATH)

if "recipes" not in st.session_state:
    st.session_state.recipes = pd.read_csv(RECIPES_PATH)

model = st.session_state.model
recipes = st.session_state.recipes

st.title("Search for similar recipes")
st.divider()


def clear_text() -> None:
    st.session_state["recipe"] = ""
    st.session_state["ingr"] = ""


recipe = st.text_area(":blue[Enter the recipe*]", key="recipe")
ingredients = st.text_area(":blue[Enter the list of ingredients*]", key="ingr")
st.button("Clear", on_click=clear_text)

st.divider()

st.header("Filtration", divider="rainbow")
st.markdown(
    "<h2 style='text-align: center; font-size: 20px;'>Do you want to set any restrictions when searching for similar recipes?</h2>",
    unsafe_allow_html=True,
)

answ1 = st.radio("Select an option:", ["YES", "NO"], index=1)
st.warning("Please note that too many restrictions can affect the quality of the search result")
st.divider()

filter_info = FilterInfo()

if answ1 == "YES":
    filter_info.methods = st.multiselect(
        "What cooking methods would you like to see in similar recipes?",
        [
            "Any",
            "Baking",
            "Freezing",
            "Boiling",
            "Raw Food",
            "Smoking",
            "Microwave",
            "Frying",
            "Stewing",
            "Sous Vide",
            "Grilling",
            "Steaming",
        ],
        ["Any"],
    )

    st.divider()
    filter_info.ingr_exclude = st.multiselect(
        "Select the foods you want to exclude: (you can choose nothing)",
        [
            "Vegetables",
            "Fruits",
            "Meat",
            "Seafood",
            "Mushrooms",
            "Dairy",
            "Grains",
            "Nuts",
        ],
    )

    st.divider()
    answ2 = st.radio("Do you want to choose a calorie range?", ["YES", "NO"], index=1)
    if answ2 == "YES":
        filter_info.calories_range = Range(
            *st.slider(
                "Select a calorie range (per servings)",
                min_value=filter_info.CALORIES_LIMITS.min,
                max_value=filter_info.CALORIES_LIMITS.max,
                value=astuple(filter_info.calories_range),
            )
        )
        st.write(
            "You selected calorie range between",
            filter_info.calories_range.min,
            "and",
            filter_info.calories_range.max,
        )

    st.divider()
    answ3 = st.radio("Do you want to choose a protein range?", ["YES", "NO"], index=1)
    if answ3 == "YES":
        filter_info.proteins_range = Range(
            *st.slider(
                "Select a protein range (per servings)",
                min_value=filter_info.PROTEINS_LIMITS.min,
                max_value=filter_info.PROTEINS_LIMITS.max,
                value=astuple(filter_info.proteins_range),
            )
        )
        st.write(
            "You selected protein range between",
            filter_info.proteins_range.min,
            "and",
            filter_info.proteins_range.max,
        )

    st.divider()
    answ4 = st.radio("Do you want to choose a fat range?", ["YES", "NO"], index=1)
    if answ4 == "YES":
        filter_info.fats_range = Range(
            *st.slider(
                "Select a fat range (per servings)",
                min_value=filter_info.FATS_LIMITS.min,
                max_value=filter_info.FATS_LIMITS.max,
                value=astuple(filter_info.fats_range),
            )
        )
        st.write(
            "You selected fat range between",
            filter_info.fats_range.min,
            "and",
            filter_info.fats_range.max,
        )

    st.divider()
    answ5 = st.radio("Do you want to choose a carbs range?", ["YES", "NO"], index=1)
    if answ5 == "YES":
        filter_info.carbs_range = Range(
            *st.slider(
                "Select a carbs range (per servings)",
                min_value=filter_info.CARBS_LIMITS.min,
                max_value=filter_info.CARBS_LIMITS.max,
                value=astuple(filter_info.carbs_range),
            )
        )
        st.write(
            "You selected carbs range between",
            filter_info.carbs_range.min,
            "and",
            filter_info.carbs_range.max,
        )

    st.divider()

    answ6 = st.radio("Do you want to set max time?", ["YES", "NO"], index=1)
    if answ6 == "YES":
        hours, mins = (
            st.number_input("Hours", step=1),
            st.number_input("Minutes", step=1),
        )
        filter_info.time = hours * 60 + mins

    st.divider()

st.header("Searching", divider="rainbow")

number: int = int(
    st.number_input("Enter the number of recipes", min_value=1, max_value=50, step=1, value=5),
)

st.markdown(
    "<h2 style='text-align: left; font-size: 20px;'>if you are sure that you have entered everything correctly, click START SEARCH</h2>",
    unsafe_allow_html=True,
)

if st.button("START SEARCH"):
    if not recipe or not ingredients:
        st.warning("Please fill in the fields: Recipe and Ingredients list")
    else:
        ind_for_filter = filter_data(filter_info)

        if recipes.shape[0] - len(ind_for_filter) < N_FILTERED_RECIPES_THRESHOLD:
            st.warning("You have set too many restrictions, search is not possible")
        else:
            recipe_info = RecipeInfo(directions=recipe, ingredients=ingredients)
            top_ind = list(model.predict(recipe_info, ind_for_filter, number))
            rec_url = recipes.loc[top_ind, "URL"].to_numpy().flatten()

            st.text("Result: ")
            for i in range(int(number)):
                st.markdown(
                    f'<a href="{rec_url[i]}" target="_blank">{rec_url[i]}</a>',
                    unsafe_allow_html=True,
                )
