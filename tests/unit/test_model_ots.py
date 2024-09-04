from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from recipe_recommendation.dataclasses import RecipeInfo
from recipe_recommendation.models import ObjectsTextSimilarity

DATA_EMBEDDING = np.array(
    [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.11, 0.11, 0.11, 0.11],
        [0.12, 0.12, 0.12, 0.12],
        [0.13, 0.13, 0.13, 0.13],
    ]
)


@pytest.fixture
def setup_model():
    instance = ObjectsTextSimilarity()
    instance.model = MagicMock()
    return instance


@pytest.fixture
def query_object():
    return RecipeInfo("recipe", "ingredients")


def test_fit(setup_model):
    instance = setup_model
    instance.model.encode.side_effect = [
        [[0.1, 0.2], [0.5, 0.6], [0.11, 0.11], [0.12, 0.12], [0.13, 0.13]],
        [[0.3, 0.4], [0.7, 0.8], [0.11, 0.11], [0.12, 0.12], [0.13, 0.13]],
    ]
    data = pd.DataFrame(
        {
            "Directions": ["recipe1", "recipe2", "recipe3", "recipe4", "recipe5", "recipe6"],
            "Ingredients": [
                "ingr11, ingr12",
                "ingr21, ingr22",
                "ingr31, ingr32",
                "ingr41, ingr42",
                "ingr51, ingr52",
                "ingr61, ingr62",
            ],
        }
    )
    instance.fit(data)

    expected_count = 2
    assert instance.model.encode.call_count == expected_count

    expected_embedding = DATA_EMBEDDING
    data_emb = instance.data_embedding
    assert np.array_equal(data_emb, expected_embedding)


@pytest.mark.parametrize(
    ("filter_ind", "top_k", "similarities"),
    [
        (np.array([0, 1, 4]), 1, [0.7, 0.8, 0.85]),
        (np.array([0, 2, 3, 4]), 3, [0.6, 0.75, 0.9, 0.65]),
        (np.array([0, 1, 2, 3, 4]), 2, [0.55, 0.45, 0.8, 0.7, 0.6]),
    ],
)
def test_predict(setup_model, query_object, filter_ind, top_k, similarities):
    instance = setup_model
    instance.model.encode.side_effect = [[0.1, 0.2], [0.3, 0.6]]
    instance.model.similarity.return_value.numpy.return_value = [np.array(similarities)]

    instance.data_embedding = DATA_EMBEDDING

    top_k_filter_ind = instance.predict(query_object, filter_ind=filter_ind, top_k=top_k)
    assert isinstance(top_k_filter_ind, np.ndarray)
    assert len(top_k_filter_ind) == top_k
    assert all(ind in filter_ind for ind in top_k_filter_ind)


def test_save_load(setup_model, tmp_path):
    instance = setup_model
    instance.data_embedding = DATA_EMBEDDING
    save_path = tmp_path / "model_embedding.npy"
    instance.save(save_path)
    loaded_instance = ObjectsTextSimilarity.load(save_path)
    assert np.array_equal(instance.data_embedding, loaded_instance.data_embedding)
