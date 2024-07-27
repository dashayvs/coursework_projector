from typing import List, Any, Set
import numpy as np
import pandas as pd
import torch
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from numpy import ndarray, dtype, signedinteger, int64
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy.typing as npt

nltk.download("punkt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class WordsComparison:
    def __init__(self) -> None:
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def _get_unique_words(self, text: str) -> Set[str]:
        words = word_tokenize(text.lower())
        unique_words = {
            self.lemmatizer.lemmatize(word)
            for word in words
            if word.isalnum() and not word.isdigit() and word not in self.stop_words
        }
        return unique_words

    def _get_num_words(self, row: pd.Series) -> int:
        num = 0
        for i, elem in enumerate(row.values):
            num += len(self.unique_query_object[i].intersection(elem))
        return num

    def fit(self, text_data: pd.DataFrame) -> None:
        self.unique_words_df = text_data.applymap(self._get_unique_words)

    def predict(self, query_object: List[str], top_k: int = 10) -> Any:
        self.unique_query_object = list(map(self._get_unique_words, query_object))
        num_match = self.unique_words_df.apply(self._get_num_words, axis=1)
        top_5_max_values = num_match.nlargest(top_k + 1)
        return top_5_max_values.index.tolist()[1:]


class TfidfSimilarity:
    def __init__(self, n_clusters: int = 10) -> None:
        self.tfidf_vectorizer = TfidfVectorizer()
        self.n_clusters: int = n_clusters

    def fit(self, text_data: pd.DataFrame) -> None:
        self.data = text_data.copy()
        self.data["Combined_Text"] = text_data.apply(lambda row: ". ".join(row.values), axis=1)
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_df=0.8,
            smooth_idf=True,
            norm="l2",
            stop_words=list(stopwords.words("english")),
        )
        word_matrix = self.word_vectorizer.fit_transform(self.data["Combined_Text"])
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=5000,
            max_df=0.8,
            smooth_idf=True,
            norm="l2",
            stop_words=list(stopwords.words("english")),
        )
        char_matrix = self.char_vectorizer.fit_transform(self.data["Combined_Text"])
        self.data_embedding = hstack((word_matrix, char_matrix))

    def predict(
        self, query_object: List[str], top_k: int = 10
    ) -> ndarray[Any, dtype[signedinteger[Any] | int64]]:
        comb_obj = [". ".join(query_object)]
        query_vector = hstack(
            (
                self.word_vectorizer.transform(comb_obj),
                self.char_vectorizer.transform(comb_obj),
            )
        )

        similarities = cosine_similarity(query_vector, self.data_embedding)
        top_k_indices = np.argsort(similarities[0])[::-1][1 : top_k + 1]

        return top_k_indices


class ObjectsTextSimilarity:
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        vectors: list[npt.NDArray[np.float32]] = [
            self.model.encode(series.values) for _, series in data.items()
        ]
        self.data_embedding = np.hstack(vectors)

    def predict(
        self,
        query_object_lst: list[str],
        top_k: int = 10,
        filtr_ind: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64]:
        query_vector = np.hstack(self.model.encode(query_object_lst))

        similarities = self.model.similarity(self.data_embedding, query_vector).view(-1).numpy()

        if filtr_ind is not None:
            similarities[filtr_ind] = -1.0

        top_k_indices = np.argsort(similarities)[: -top_k - 1 : -1]

        return top_k_indices


class ObjectsSimilarityFiltered:
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.filter_data = pd.read_csv("D:\\recipe_recomendation\\data\\filter_data_recipes.csv")

    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        vectors: list[npt.NDArray[np.float32]] = [
            self.model.encode(series.values) for _, series in data.items()
        ]
        self.data_embedding = np.hstack(vectors)

    def _filter(self, row: pd.Series) -> int:
        relevant_columns = row.index.intersection(self.filter_features.index)
        return int((row[relevant_columns] == self.filter_features).sum())

    def predict(
        self,
        query_object_lst: List[str],
        filter_features: pd.Series,
        top_k: int = 10,
        similarity_threshold: float = 0.8,
        w1: float = 0.6,
        w2: float = 0.4,
    ) -> npt.NDArray[np.int64]:
        self.filter_features = filter_features

        query_vector = np.hstack(self.model.encode(query_object_lst))
        similarities = self.model.similarity(self.data_embedding, query_vector).view(-1).numpy()

        # Filter indices based on similarity threshold
        filtered_indices = np.where(
            (similarity_threshold <= similarities) & (similarities <= 0.98)
        )[0]

        # If there are not enough similar objects, return the available ones
        if len(filtered_indices) < top_k:
            return np.argsort(similarities)[: -top_k - 1 : -1]

        # Compute matches for filtered indices
        matches = self.filter_data.iloc[filtered_indices].apply(self._filter, axis=1)

        combined_scores = w1 * similarities[filtered_indices] + w2 * (
            matches / len(filter_features)
        )

        sorted_indices = filtered_indices[np.argsort(combined_scores)[::-1]]

        return np.array(sorted_indices[:top_k].astype(np.int64))
