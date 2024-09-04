import numpy as np
import pandas as pd
import pytest

from recipe_recommendation.dataclasses import RecipeInfo
from recipe_recommendation.models import (
    TfidfSimilarity,
    WordsComparison,
)
from recipe_recommendation.paths import RECIPES_PATH


# Tests for WordsComparison
@pytest.fixture
def model_words_comparison():
    return WordsComparison()


def test_get_num_words(model_words_comparison):
    unique_words_df = pd.DataFrame(
        {
            "Directions": [{"recipe", "direction"}, {"instruction", "cook", "apple", "pie"}],
            "Ingredients": [{"meat", "chicken", "lemon"}, {"orange", "fish", "apple"}],
        }
    )
    model_words_comparison.unique_query_object = [{"recipe", "ingredient"}, {"lemon", "water"}]
    num_words = model_words_comparison._get_num_words(unique_words_df.iloc[0])
    expected_num_words = 2
    assert num_words == expected_num_words


def test_fit_words_comparison(model_words_comparison):
    data = pd.DataFrame(
        {
            "Directions": [
                "Some recipe recipe, directions",
                "Instruction, how to cook apple apple pie",
            ],
            "Ingredients": ["meat, chicken, meat, lemon", "apple, orange, fish, fish"],
        }
    )
    model_words_comparison.fit(data)
    expected_unique_words_df = pd.DataFrame(
        {
            "Directions": [{"recipe", "direction"}, {"instruction", "cook", "apple", "pie"}],
            "Ingredients": [{"meat", "chicken", "lemon"}, {"orange", "fish", "apple"}],
        }
    )
    assert model_words_comparison.unique_words_df.equals(expected_unique_words_df)


def test_predict_words_comparison(model_words_comparison):
    model_words_comparison.unique_words_df = pd.DataFrame(
        {
            "Directions": [
                {"recipe", "direction"},
                {"instruction", "cook", "apple", "pie"},
                {"boil", "fry", "grill"},
                {"prepare", "chop", "season"},
            ],
            "Ingredients": [
                {"meat", "chicken", "lemon"},
                {"orange", "fish", "apple"},
                {"salt", "pepper", "garlic"},
                {"onion", "tomato", "cucumber"},
            ],
        }
    )

    query_object = RecipeInfo(
        directions="recipe cook pie apple instructions boil",
        ingredients="apple pie cinnamon sugar salt lemon",
    )

    ind = model_words_comparison.predict(query_object, 3)
    expected_ind = np.array([0, 1, 2])
    assert np.all(np.isin(expected_ind, ind)) and np.all(np.isin(ind, expected_ind))


def test_save_and_load_words_comparison(model_words_comparison, tmp_path):
    model_words_comparison.unique_words_df = pd.DataFrame(
        {
            "Directions": [{"recipe", "direction"}, {"instruction", "cook", "apple", "pie"}],
            "Ingredients": [{"meat", "chicken", "lemon"}, {"orange", "fish", "apple"}],
        }
    )
    file_path = tmp_path / "unique_words_df.csv"
    model_words_comparison.save(file_path)
    assert file_path.exists()

    loaded_model = WordsComparison.load(file_path)
    assert loaded_model.unique_words_df.equals(model_words_comparison.unique_words_df)


#
# Tests for TfidfSimilarity
#
@pytest.fixture
def sample_data():
    return pd.read_csv(RECIPES_PATH).iloc[:5, :2]


@pytest.fixture
def query_object():
    return RecipeInfo("Sample direction. recipe", "Sample ingredients. lemon")


@pytest.fixture
def model_tfidf_similarity():
    return TfidfSimilarity()


def test_fit_tfidf_similarity(model_tfidf_similarity, sample_data):
    model_tfidf_similarity.fit(sample_data)
    assert model_tfidf_similarity.data_embedding.shape[0] == sample_data.shape[0]
    assert model_tfidf_similarity.data_embedding.shape[1] > 0


@pytest.mark.parametrize("top_k", (1, 2, 3, 4))
def test_predict_tfidf_similarity(model_tfidf_similarity, sample_data, query_object, top_k):
    model_tfidf_similarity.fit(sample_data)

    top_k_indices = model_tfidf_similarity.predict(query_object, top_k=top_k)
    assert len(top_k_indices) == top_k


def test_save_and_load_tfidf_similarity(model_tfidf_similarity, tmp_path):
    data = pd.DataFrame(
        {
            "Directions": ["Mix the ingredients well.", "Boil water."],
            "Ingredients": ["flour sugar eggs", "water salt"],
        }
    )
    model_tfidf_similarity.fit(data)
    file_path = tmp_path / "data_embedding.npy"
    model_tfidf_similarity.save(file_path)
    assert file_path.exists()

    loaded_model = TfidfSimilarity.load(file_path)
    assert np.array_equal(loaded_model.data_embedding, model_tfidf_similarity.data_embedding)


if __name__ == "__main__":
    pytest.main()
