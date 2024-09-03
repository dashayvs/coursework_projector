import numpy as np
import pandas as pd
import pytest

from recipe_recommendation.dataclasses import RecipeInfo
from recipe_recommendation.models import ObjectsTextSimilarity
from recipe_recommendation.paths import RECIPES_PATH


@pytest.fixture
def sample_data():
    data = pd.read_csv(RECIPES_PATH).iloc[:5, :2]
    return data


@pytest.fixture
def model():
    return ObjectsTextSimilarity()


@pytest.fixture
def query_object():
    return RecipeInfo("Sample direction.", "Sample ingredients.")


@pytest.fixture
def filter_ind():
    return np.array([0, 1, 2, 3, 4])


def test_fit(model, sample_data):
    model.fit(sample_data)
    assert model.data_embedding.shape == (5, 2 * 384)


@pytest.mark.parametrize(
    ("filter_ind", "top_k"),
    [(np.array([0, 1, 4]), 1), (np.array([0, 2, 3, 4]), 3), (np.array([0, 1, 2, 3, 4]), 2)],
)
def test_predict(model, sample_data, query_object, filter_ind, top_k):
    model.fit(sample_data)
    top_k_filter_ind = model.predict(query_object, filter_ind, top_k=top_k)
    assert len(top_k_filter_ind) == top_k
    assert all(ind in filter_ind for ind in top_k_filter_ind)


def test_save_and_load(model, sample_data, tmp_path):
    model.fit(sample_data)
    file_path = tmp_path / "embedding.npy"
    model.save(file_path)
    assert file_path.exists()

    loaded_model = ObjectsTextSimilarity.load(file_path)
    assert np.array_equal(loaded_model.data_embedding, model.data_embedding)
