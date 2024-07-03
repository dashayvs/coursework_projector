# bestmodel.py
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ObjectsTextSimilarity:
    def __init__(self) -> None:
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        self.name_text_features = self.data.columns
        vectors = []
        for name in self.name_text_features:
            text_feature = list((self.data.loc[:, [name]]).values.flatten())
            vectors.append(self.model.encode(text_feature, convert_to_tensor=True).to(device))
        self.data_embedding = torch.cat(vectors, dim=1)

    def predict(self, query_object_lst: List[str], top_k: int = 10,
                filtr_ind: Optional[np.ndarray] = None) -> np.ndarray:
        vectors = self.model.encode(query_object_lst, convert_to_tensor=True).to(device)
        query_vector = vectors.view(-1)
        similarities = cosine_similarity(query_vector, self.data_embedding).cpu().numpy()
        if filtr_ind is not None:
            similarities[filtr_ind] = -1
        top_k_indices = np.argsort(similarities)[::-1][1:top_k + 1]

        return top_k_indices
