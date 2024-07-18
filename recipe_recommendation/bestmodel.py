# bestmodel.py
from typing import List, Optional, Any, cast
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class ObjectsTextSimilarity:
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        self.name_text_features = self.data.columns
        text_features: List[List[str]] = [list(data[col].values.flatten()) for col in data.columns]
        vectors: List[torch.Tensor] = [
            cast(torch.Tensor, self.model.encode(text_feature, device=device))
            for text_feature in text_features
        ]
        self.data_embedding = torch.cat(vectors, dim=1)

    def predict(
        self,
        query_object_lst: List[str],
        top_k: int = 10,
        filtr_ind: Optional[np.ndarray[Any, np.dtype[Any]]] = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        vectors = cast(torch.Tensor, self.model.encode(query_object_lst, device=device))
        query_vector = vectors.view(-1)
        similarities = cosine_similarity(
            query_vector.cpu().numpy().reshape(1, -1), self.data_embedding.cpu().numpy()
        )
        if filtr_ind is not None:
            similarities[0, filtr_ind] = -1
        top_k_indices = np.argsort(similarities[0])[::-1][:top_k]

        return top_k_indices
