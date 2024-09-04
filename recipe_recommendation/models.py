from abc import ABC, abstractmethod
from os import PathLike
from typing import Any, Self

import nltk
import numpy as np
import numpy.typing as npt
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recipe_recommendation.dataclasses import RecipeInfo
from recipe_recommendation.paths import FILTER_DATA_PATH


class ModelTemplate(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None: ...

    @abstractmethod
    def predict(
        self, query_object: RecipeInfo, *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.int64]: ...

    @abstractmethod
    def save(self, path: PathLike[str]) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: PathLike[str]) -> Self: ...


class WordsComparison(ModelTemplate):
    def __init__(self) -> None:
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english"))

    def _get_unique_words(self, text: str) -> set[str]:
        words = word_tokenize(text.lower())
        unique_words = {
            self.lemmatizer.lemmatize(word)
            for word in words
            if word.isalnum() and not word.isdigit() and word not in self.stop_words
        }
        return unique_words

    def _get_num_words(self, row: "pd.Series[float]") -> int:
        return sum(
            len(self.unique_query_object[i].intersection(elem)) for i, elem in enumerate(row.values)
        )

    def fit(self, data: pd.DataFrame) -> None:
        self.unique_words_df = data.map(self._get_unique_words)

    def predict(self, query_object: RecipeInfo, top_k: int = 10) -> npt.NDArray[np.int64]:
        query_object_lst = [query_object.directions, query_object.ingredients]
        self.unique_query_object = list(map(self._get_unique_words, query_object_lst))
        num_match = self.unique_words_df.apply(self._get_num_words, axis=1)
        top_5_max_values = num_match.nlargest(top_k)
        return np.array(top_5_max_values.index)

    def save(self, path: PathLike[str]) -> None:
        unique_words_df = self.unique_words_df.copy()
        unique_words_df = unique_words_df.map(
            lambda x: ",".join(sorted(x)) if isinstance(x, set) else x
        )
        unique_words_df.to_csv(path, index=False)

    @classmethod
    def load(cls, path: PathLike[str]) -> Self:
        unique_words_df = pd.read_csv(path)
        unique_words_df = unique_words_df.map(lambda x: set(x.split(",")) if "," in x else x)
        model = cls()
        model.unique_words_df = unique_words_df
        return model


class TfidfSimilarity(ModelTemplate):
    def __init__(self) -> None:
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            smooth_idf=True,
            max_df=0.8,
            norm="l2",
            stop_words=list(stopwords.words("english")),
        )

        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=5000,
            max_df=0.8,
            smooth_idf=True,
            norm="l2",
            stop_words=list(stopwords.words("english")),
        )

    def fit(self, data: pd.DataFrame) -> None:
        combined_text = data.agg(" ".join, axis=1).tolist()

        word_matrix = self.word_vectorizer.fit_transform(combined_text)
        char_matrix = self.char_vectorizer.fit_transform(combined_text)

        self.data_embedding = hstack((word_matrix, char_matrix)).toarray()

    def predict(self, query_object: RecipeInfo, top_k: int = 10) -> npt.NDArray[np.int64]:
        combined_query_object = [". ".join([query_object.directions, query_object.ingredients])]

        word_matrix = self.word_vectorizer.transform(combined_query_object)
        char_matrix = self.char_vectorizer.transform(combined_query_object)

        query_vector = hstack((word_matrix, char_matrix)).toarray()

        [similarities] = cosine_similarity(query_vector, self.data_embedding)
        top_k_indices = np.argsort(similarities)[: -top_k - 1 : -1]
        return top_k_indices

    def save(self, path: PathLike[str]) -> None:
        np.save(path, self.data_embedding)

    @classmethod
    def load(cls, path: PathLike[str]) -> Self:
        model = cls()
        model.data_embedding = np.load(path)
        return model


# todo unit tests
class ObjectsTextSimilarity(ModelTemplate):
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.duplicate_threshold = 0.98

    def fit(self, data: pd.DataFrame) -> None:
        vectors: list[npt.NDArray[np.float32]] = [
            self.model.encode(series.values) for _, series in data.items()
        ]
        self.data_embedding = np.hstack(vectors)

    def predict(
        self,
        query_object: RecipeInfo,
        filter_ind: npt.NDArray[np.int64],
        top_k: int = 10,
    ) -> npt.NDArray[np.int64]:
        query_vector = np.hstack(
            self.model.encode([query_object.directions, query_object.ingredients]),
        )
        [similarities] = self.model.similarity(
            query_vector, self.data_embedding[filter_ind]
        ).numpy()
        # delete recipe which is equal to query_object
        similarities[similarities > self.duplicate_threshold] = -1.0
        top_k_indices = np.argsort(similarities)[: -top_k - 1 : -1]
        top_k_filter_ind: npt.NDArray[np.int64] = filter_ind[top_k_indices]
        return top_k_filter_ind

    def save(self, path: PathLike[str]) -> None:
        np.save(path, self.data_embedding)

    @classmethod
    def load(cls, path: PathLike[str]) -> Self:
        model = cls()
        model.data_embedding = np.load(path)
        return model


class ObjectsSimilarityFiltered(ModelTemplate):
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.duplicate_threshold = 0.95
        self.filter_data = pd.read_csv(FILTER_DATA_PATH)

    def fit(self, data: pd.DataFrame) -> None:
        vectors: list[npt.NDArray[np.float32]] = [
            self.model.encode(series.values) for _, series in data.items()
        ]
        self.data_embedding = np.hstack(vectors)

    def _filter(self, row: "pd.Series[float]", filter_features: "pd.Series[float]") -> float:
        return (row[filter_features.index] == filter_features).sum()

    def predict(
        self,
        query_object: RecipeInfo,
        filter_features: "pd.Series[float] | None" = None,
        top_k: int = 10,
        similarity_threshold: float = 0.8,
        w: float = 0.6,
    ) -> npt.NDArray[np.int64]:
        if filter_features is None:
            raise ValueError("filter_features cannot be None")

        query_vector = np.hstack(
            self.model.encode([query_object.directions, query_object.ingredients]),
        )

        [similarities] = self.model.similarity(query_vector, self.data_embedding).numpy()

        # Filter indices based on similarity threshold
        filtered_indices = np.where(
            (similarity_threshold <= similarities) & (similarities <= self.duplicate_threshold),
        )[0]

        # If there are not enough similar objects, return the available ones
        if len(filtered_indices) < top_k:
            return np.argsort(similarities)[: -top_k - 1 : -1]

        # Compute matches for filtered indices
        matches = self.filter_data.iloc[filtered_indices].apply(
            self._filter,
            args=(filter_features,),
            axis=1,
        )

        combined_scores = w * similarities[filtered_indices] + (1.0 - w) * (
            matches / len(filter_features)
        )

        sorted_indices = filtered_indices[np.argsort(combined_scores)[: -top_k - 1 : -1]]

        return np.array(sorted_indices[:top_k].astype(np.int64))

    def save(self, path: PathLike[str]) -> None:
        np.save(path, self.data_embedding)

    @classmethod
    def load(cls, path: PathLike[str]) -> Self:
        model = cls()
        model.data_embedding = np.load(path)
        return model
