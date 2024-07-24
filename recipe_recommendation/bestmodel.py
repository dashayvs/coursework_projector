# bestmodel.py
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy.typing as npt


class ObjectsTextSimilarity:
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        vectors: list[npt.NDArray[np.float32]] = [
            self.model.encode(series.values, convert_to_numpy=True, normalize_embeddings=True)
            for _, series in data.items()
        ]
        self.data_embedding = np.hstack(vectors)

    def predict(
        self,
        query_object_lst: list[str],
        top_k: int = 10,
        filtr_ind: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        query_vector = np.hstack(
            self.model.encode(query_object_lst, convert_to_numpy=True, normalize_embeddings=True)
        )

        # if isinstance(self.data_embedding, torch.Tensor):
        #     self.data_embedding = self.data_embedding.cpu().numpy()
        #     print(1)
        # if isinstance(query_vector, torch.Tensor):
        #     query_vector = query_vector.cpu().numpy()
        #     print(2)

        similarities = self.model.similiarity(self.data_embedding, query_vector).numpy()

        if filtr_ind is not None:
            similarities[filtr_ind] = -1.0

        top_k_indices = np.argsort(similarities)[: -top_k - 1 : -1]

        return top_k_indices
