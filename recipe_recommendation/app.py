from typing import List, Any
import pandas as pd
import streamlit as st
from pathlib import Path
from recipe_recommendation.filter import filter_data
from recipe_recommendation.recipe_info import RecipeInfo
from recipe_recommendation.models import ObjectsTextSimilarity

ROOT_DIR = Path(__file__).parent.parent

MODEL_PATH = ROOT_DIR / "model" / "ObjectsTextSimilarityModel1.npy"
RECIPES_PATH = ROOT_DIR / "data" / "train_data_text_url.csv"

# todo create streamlit const
model = ObjectsTextSimilarity.load(MODEL_PATH)
recipes = pd.read_csv(RECIPES_PATH)


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

param_list: List[List[Any]] = [[]] * 7

if answ1 == "YES":
    param_list[0] = st.multiselect(
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
    param_list[1] = st.multiselect(
        "Select the foods you want to exclude: " "(you can choose nothing)",
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
        param_list[2] = st.slider(
            "Select a calorie range (per servings)",
            min_value=0,
            max_value=1000,
            value=(0, 1000),
        )
        st.write(
            "You selected calorie range between",
            param_list[2][0],
            "and",
            param_list[2][1],
        )

    st.divider()
    answ3 = st.radio("Do you want to choose a protein range?", ["YES", "NO"], index=1)
    if answ3 == "YES":
        param_list[3] = st.slider(
            "Select a protein range (per servings)",
            min_value=0,
            max_value=40,
            value=(0, 40),
        )
        st.write(
            "You selected protein range between",
            param_list[3][0],
            "and",
            param_list[3][1],
        )

    st.divider()
    answ4 = st.radio("Do you want to choose a fat range?", ["YES", "NO"], index=1)
    if answ4 == "YES":
        param_list[4] = st.slider(
            "Select a fat range (per servings)",
            min_value=0,
            max_value=50,
            value=(0, 50),
        )
        st.write("You selected fat range between", param_list[4][0], "and", param_list[4][1])

    st.divider()
    answ5 = st.radio("Do you want to choose a carbs range?", ["YES", "NO"], index=1)
    if answ5 == "YES":
        param_list[5] = st.slider(
            "Select a carbs range (per servings)",
            min_value=0,
            max_value=100,
            value=(0, 100),
        )
        st.write("You selected fat range between", param_list[5][0], "and", param_list[5][1])

    st.divider()

    answ6 = st.radio("Do you want to set max time?", ["YES", "NO"], index=1)
    if answ6 == "YES":
        hours, mins = (
            st.number_input("Hours", step=1),
            st.number_input("Minutes", step=1),
        )
        param_list[6] = [hours * 60 + mins]

    st.divider()

st.header("Searching", divider="rainbow")

number: int | float = st.number_input(
    "Enter the number of recipes", min_value=1, max_value=50, step=1, value=5
)

st.markdown(
    "<h2 style='text-align: left; font-size: 20px;'>if you are sure that you have entered everything correctly, click START SEARCH</h2>",
    unsafe_allow_html=True,
)

if st.button("START SEARCH"):
    if not recipe or not ingredients:
        st.warning("Please fill in the fields: Recipe and Ingredients list")
    else:
        ind_for_filter = filter_data(param_list)
        if ind_for_filter == 0:
            st.warning("You have set too many restrictions, search is not possible")
        else:
            recipe_info = RecipeInfo(directions=recipe, ingredients=ingredients)
            top_ind = list(model.predict(recipe_info, number, ind_for_filter))
            rec_url = recipes.iloc[top_ind, :].loc[:, ["URL"]].values.flatten()

            st.text("Result: ")
            for i in range(int(number)):
                st.markdown(
                    f'<a href="{rec_url[i]}" target="_blank">{rec_url[i]}</a>',
                    unsafe_allow_html=True,
                )
