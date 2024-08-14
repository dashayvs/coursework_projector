from typing import Self, AnyStr
import numpy as np
import pandas as pd
import torch
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy.typing as npt
from recipe_recommendation.recipe_info import RecipeInfo
from os import PathLike

nltk.download("punkt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class WordsComparison:
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

    def _get_num_words(self, row: pd.Series) -> int:
        return sum(
            len(self.unique_query_object[i].intersection(elem)) for i, elem in enumerate(row.values)
        )

    def fit(self, text_data: pd.DataFrame) -> None:
        self.unique_words_df = text_data.map(self._get_unique_words)

    def predict(self, query_object: list[str], top_k: int = 10) -> npt.NDArray[np.int64]:
        self.unique_query_object = list(map(self._get_unique_words, query_object))
        num_match = self.unique_words_df.apply(self._get_num_words, axis=1)
        top_5_max_values = num_match.nlargest(top_k)
        return np.array(top_5_max_values.index)


class TfidfSimilarity:
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

    def fit(self, text_data: pd.DataFrame) -> None:
        combined_text = text_data.agg(" ".join, axis=1).values.tolist()

        word_matrix = self.word_vectorizer.fit_transform(combined_text)
        char_matrix = self.char_vectorizer.fit_transform(combined_text)

        self.data_embedding = hstack((word_matrix, char_matrix))

    def predict(self, query_object: list[str], top_k: int = 10) -> npt.NDArray[np.int64]:
        combined_query_object = [". ".join(query_object)]

        word_matrix = self.word_vectorizer.transform(combined_query_object)
        char_matrix = self.char_vectorizer.transform(combined_query_object)

        query_vector = hstack((word_matrix, char_matrix))

        similarities = cosine_similarity(query_vector, self.data_embedding)
        top_k_indices = np.argsort(similarities[0])[: -top_k + 1 : -1]
        return top_k_indices


# todo unit tests
class ObjectsTextSimilarity:
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.duplicate_threshold = 0.98

    def fit(self, data: pd.DataFrame) -> None:
        vectors: list[npt.NDArray[np.float32]] = [
            self.model.encode(series.values) for _, series in data.items()
        ]
        self.data_embedding = np.hstack(vectors)

    def predict(
        self, query_object: RecipeInfo, filtr_ind: npt.NDArray[np.int64], top_k: int = 10
    ) -> npt.NDArray[np.int64]:
        query_vector = np.hstack(
            self.model.encode([query_object.directions, query_object.ingredients])
        )
        # todo apply filter before doing similarity
        [similarities] = self.model.similarity(query_vector, self.data_embedding).numpy()
        # delete recipe which is equal to query_object
        similarities[similarities > self.duplicate_threshold] = -1.0
        similarities[filtr_ind] = -1.0
        top_k_indices = np.argsort(similarities)[: -top_k - 1 : -1]
        return top_k_indices

    def save(self, path: str) -> None:
        np.save(path, self.data_embedding)

    @classmethod
    def load(cls, path: PathLike[AnyStr]) -> Self:
        model = cls()
        model.data_embedding = np.load(path)
        return model


class ObjectsSimilarityFiltered:
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # todo change path
        self.filter_data = pd.read_csv("data\\filter_data_recipes.csv")
        self.duplicate_threshold = 0.98

    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        vectors: list[npt.NDArray[np.float32]] = [
            self.model.encode(series.values) for _, series in data.items()
        ]
        self.data_embedding = np.hstack(vectors)

    def _filter(self, row: pd.Series, filter_features: pd.Series) -> int:
        return int((row[filter_features.index] == filter_features).sum())

    def predict(
        self,
        query_object_lst: list[str],
        filter_features: pd.Series,
        top_k: int = 10,
        similarity_threshold: float = 0.8,
        w: float = 0.6,
    ) -> npt.NDArray[np.int64]:
        self.filter_features = filter_features

        query_vector = np.hstack(self.model.encode(query_object_lst))
        [similarities] = self.model.similarity(query_vector, self.data_embedding).numpy()

        # Filter indices based on similarity threshold
        filtered_indices = np.where(
            (similarity_threshold <= similarities) & (similarities <= self.duplicate_threshold)
        )[0]

        # If there are not enough similar objects, return the available ones
        if len(filtered_indices) < top_k:
            return np.argsort(similarities)[: -top_k - 1 : -1]

        # Compute matches for filtered indices
        matches = self.filter_data.iloc[filtered_indices].apply(
            self._filter, args=(filter_features,), axis=1
        )

        combined_scores = w * similarities[filtered_indices] + (1.0 - w) * (
            matches / len(filter_features)
        )

        sorted_indices = filtered_indices[np.argsort(combined_scores)[: -top_k - 1 : -1]]

        return np.array(sorted_indices[:top_k].astype(np.int64))
