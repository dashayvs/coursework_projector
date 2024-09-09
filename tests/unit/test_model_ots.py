from unittest.mock import create_autospec, patch

import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

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


def patch_init(self):
    self.model = create_autospec(SentenceTransformer)
    self.duplicate_threshold = 0.98


@pytest.fixture
def model():
    with patch("recipe_recommendation.models.SentenceTransformer"):
        return ObjectsTextSimilarity()


@pytest.fixture
def query_object():
    return RecipeInfo("recipe", "ingredients")


@patch("recipe_recommendation.models.SentenceTransformer", create_autospec(SentenceTransformer))
def test_fit():
    model = ObjectsTextSimilarity()
    model.model.encode.side_effect = [
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
    model.fit(data)

    assert model.model.encode.call_count == data.shape[1]

    expected_embedding = DATA_EMBEDDING
    data_emb = model.data_embedding
    assert np.array_equal(data_emb, expected_embedding)


@patch("recipe_recommendation.models.SentenceTransformer", create_autospec(SentenceTransformer))
@pytest.mark.parametrize(
    ("filter_ind", "top_k", "similarities"),
    [
        (np.array([0, 1, 4]), 1, [0.7, 0.8, 0.85]),
        (np.array([0, 2, 3, 4]), 3, [0.6, 0.75, 0.9, 0.65]),
        (np.array([0, 1, 2, 3, 4]), 2, [0.55, 0.45, 0.8, 0.7, 0.6]),
    ],
)
def test_predict(query_object, filter_ind, top_k, similarities):
    model = ObjectsTextSimilarity()
    model.model.encode.side_effect = [[0.1, 0.2], [0.3, 0.6]]
    model.model.similarity.return_value.numpy.return_value = [np.array(similarities)]

    model.data_embedding = DATA_EMBEDDING

    top_k_filter_ind = model.predict(query_object, filter_ind=filter_ind, top_k=top_k)
    assert isinstance(top_k_filter_ind, np.ndarray)
    assert len(top_k_filter_ind) == top_k
    assert all(ind in filter_ind for ind in top_k_filter_ind)


def test_save_load(model, tmp_path):
    model.data_embedding = DATA_EMBEDDING
    save_path = tmp_path / "model_embedding.npy"
    model.save(save_path)

    with patch.object(ObjectsTextSimilarity, "__init__", patch_init):
        loaded_instance = ObjectsTextSimilarity.load(save_path)
    assert np.array_equal(model.data_embedding, loaded_instance.data_embedding)


if __name__ == "__main__":
    pytest.main()
