import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from recipe_recommendation.dataclasses import RecipeInfo
from recipe_recommendation.models import ObjectsSimilarityFiltered
from recipe_recommendation.paths import BEST_MODEL_PATH, FILTER_DATA_PATH

DATA_EMBEDDING = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
logger = logging.getLogger("model")

# def patch_init(self):
#     self.model = create_autospec(SentenceTransformer)
#     self.duplicate_threshold = 0.98
#
#
# @pytest.fixture
# def model():
#     with patch.object(ObjectsTextSimilarity, "__init__", patch_init):
#         return ObjectsTextSimilarity()


@pytest.fixture
def setup_model():
    filter_data = pd.DataFrame(
        {
            "feature1": [0.1, 0.2],
            "feature2": [0.2, 0.3],
            "feature3": ["a", "b"],
        }
    )

    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = filter_data
        instance = ObjectsSimilarityFiltered()
    instance.model = MagicMock()
    return instance


@pytest.fixture
def query_object():
    return RecipeInfo("recipe", "ingredients")


def test_fit(setup_model):
    instance = setup_model
    instance.model.encode.side_effect = [[[0.1, 0.2], [0.5, 0.6]], [[0.3, 0.4], [0.7, 0.8]]]
    data = pd.DataFrame(
        {"Directions": ["recipe1", "recipe2"], "Ingredients": ["ingr11, ingr12", "ingr21, ingr22"]}
    )
    instance.fit(data)

    expected_count = 2
    assert instance.model.encode.call_count == expected_count

    expected_embedding = DATA_EMBEDDING
    data_emb = instance.data_embedding
    assert np.array_equal(data_emb, expected_embedding)


@pytest.mark.parametrize(
    ("data", "filter_features", "expected_result"),
    [
        ([1, 0, "Meat", "Soup", 25], [1, 1, "Meat", 0], 2),
        ([2, 1, "Vegetable", "Salad", 30], [2, 1, "Vegetable", 0], 3),
        ([1, 0, "Fish", "Stew", 20], [1, 1, "Fish", 0], 2),
        ([3, 2, "Chicken", "Curry", 0], [3, 2, "Chicken", 0], 4),
        ([4, 3, "Beef", "Burger", 35], [1, 0, "Meat", 25], 0),
    ],
)
def test_filter_objects_similarity_filtered(setup_model, data, filter_features, expected_result):
    instance = setup_model
    data = pd.Series(data, index=["category1", "category2", "category3", "category4", "category5"])
    filter_features = pd.Series(
        filter_features, index=["category1", "category2", "category3", "category5"]
    )

    assert instance._filter(data, filter_features) == expected_result


def test_predict_no_filter_features(setup_model, query_object):
    instance = setup_model
    with pytest.raises(ValueError, match="filter_features cannot be None"):
        instance.predict(query_object, filter_features=None)


def test_predict_with_filter_features(setup_model, query_object):
    instance = setup_model
    instance.model.encode.side_effect = [[0.1, 0.2], [0.3, 0.6]]
    instance.model.similarity.return_value.numpy.return_value = [np.array([0.85, 0.81])]

    instance.data_embedding = DATA_EMBEDDING
    filter_features = pd.Series([0.1, 0.2, "b"], index=["feature1", "feature2", "feature3"])

    result = instance.predict(query_object, filter_features=filter_features, top_k=1)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result[0] == 0


def test_save_load(setup_model, tmp_path):
    instance = setup_model
    instance.data_embedding = DATA_EMBEDDING
    save_path = tmp_path / "model_embedding.npy"
    instance.save(save_path)
    logger.info(FILTER_DATA_PATH)
    logger.info(BEST_MODEL_PATH)
    loaded_instance = ObjectsSimilarityFiltered.load(save_path)

    assert np.array_equal(instance.data_embedding, loaded_instance.data_embedding)


if __name__ == "__main__":
    pytest.main()
