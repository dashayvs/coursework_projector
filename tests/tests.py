import unittest
import sys
import os
from unittest.mock import patch

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recipe_recommendation.feature_eng_funcs import *


class TestFeatureEngineering(unittest.TestCase):
    def test_clean_categories0(self):
        self.assertEqual(clean_categories0("Dinner Recipes, Vegan, Vegetarian Recipes, Breakfast"),
                         "Dinner, Vegan, Vegetarian, Breakfast")
        self.assertEqual(clean_categories0("Recipes, Soup Recipes, Vegan"), "Soup, Vegan")
        self.assertEqual(clean_categories0("Lunch Recipes, Dinner Recipes, , Dessert"), "Lunch, Dinner, Dessert")

    def test_clean_categories1(self):
        self.assertEqual(set(clean_categories1("Dinner, Vegan, Dinner, Breakfast and Lunch").split(', ')),
                         {"Breakfast", "Dinner", "Lunch", "Vegan"})
        self.assertEqual(set(clean_categories1("Soup, Vegan, Soup, Vegan").split(', ')), {"Soup", "Vegan"})


    def test_singular_to_plural(self):
        self.assertEqual(set(singular_to_plural("Dinner, Vegan, Breakfast Recipe").split(', ')),
                         {"Dinner", "Vegan", "Breakfast Recipes"})
        self.assertEqual(set(singular_to_plural("Soup Recipe, Vegan").split(', ')), {"Soup Recipes", "Vegan"})

    def test_generate_combinations(self):
        p = inflect.engine()
        combinations_list = generate_combinations("Dinner", "Recipe", p)
        self.assertIn("Dinner Recipe", combinations_list)
        self.assertIn("Recipes Dinner", combinations_list)
        self.assertIn("Dinners Recipes", combinations_list)

    @patch('recipe_recommendation.feature_eng_funcs.generate_combinations')
    def test_clean_categories3(self, mock_generate_combinations):
        mock_generate_combinations.side_effect = lambda word1, word2, p: [
            f"{word1} {word2}",
            f"{p.plural(word1)} {word2}",
            f"{word1} {p.plural(word2)}",
            f"{p.plural(word1)} {p.plural(word2)}"
        ]

        self.assertEqual(set(clean_categories3("Apple Pie, Apple, Pie, Dessert").split(', ')),
                         {"Apple", "Pie", "Dessert"})


    def test_get_meal(self):
        self.assertEqual(get_meal("Dinner, Vegan"), "Dinner")
        self.assertEqual(get_meal("Lunch, Vegan"), "Lunch")
        self.assertEqual(get_meal("Soup, Vegan"), "None")

    def test_get_course(self):
        self.assertEqual(get_course("Main Dish, Vegan, Breakfast"), "Main Dish")
        self.assertEqual(get_course("Side Dish, Vegan"), "Side Dish")
        self.assertEqual(get_course("Apple, Vegan"), "None")

    def test_vegan_vegetarian(self):
        self.assertEqual(vegan_vegetarian("Dinner, Vegan, Breakfast"), "Vegan")
        self.assertEqual(vegan_vegetarian("Lunch, Vegetarian"), "Vegetarian")
        self.assertEqual(vegan_vegetarian("Soup, Meat"), "None")

    def test_convert_time_to_minutes(self):
        self.assertEqual(convert_time_to_minutes("1 hr 30 mins"), 90)
        self.assertEqual(convert_time_to_minutes("2 hrs 15 mins"), 135)

    def test_fill_cat_ingr(self):
        data = pd.DataFrame({'name': ['test1', 'test2'], 'tomato': [0, 0], 'potato': [0, 0]})
        result_ingr = [['tomato'], ['potato']]
        data = data.apply(lambda row: fill_cat_ingr(row, result_ingr), axis=1)
        self.assertEqual(data.loc[0, 'tomato'], 1)
        self.assertEqual(data.loc[1, 'potato'], 1)


if __name__ == '__main__':
    unittest.main()
