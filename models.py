import numpy as np
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

nltk.download('punkt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WordsComparison:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def _get_unique_words(self, text):
        words = nltk.word_tokenize(text.lower())
        unique_words = [self.lemmatizer.lemmatize(word) for word in words if
                        word.isalnum() and not word.isdigit() and word not in self.stop_words]
        return set(unique_words)

    def _get_num_words(self, row):
        num = 0
        for i, elem in enumerate(row.values):
            num += len(self.unique_query_object[i].intersection(elem))
        return num

    def fit(self, text_data):
        self.unique_words_df = text_data.applymap(self._get_unique_words)

    def predict(self, query_object, top_k=10):
        self.unique_query_object = list(map(self._get_unique_words, query_object))
        num_match = self.unique_words_df.apply(self._get_num_words, axis=1)
        top_5_max_values = num_match.nlargest(top_k + 1)
        return top_5_max_values.index.tolist()[1:]


class TfidfSimilarity:
    def __init__(self, n_clusters=10):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.n_clusters = n_clusters

    def fit(self, text_data):
        self.data = text_data.copy()
        self.data['Combined_Text'] = text_data.apply(lambda row: '. '.join(row.values), axis=1)
        self.word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_df=0.8, smooth_idf=True,
                                               norm='l2',
                                               stop_words=list(stopwords.words('english')))
        word_matrix = self.word_vectorizer.fit_transform(self.data['Combined_Text'])
        self.char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000, max_df=0.8,
                                               smooth_idf=True, norm='l2', stop_words=list(stopwords.words('english')))
        char_matrix = self.char_vectorizer.fit_transform(self.data['Combined_Text'])
        self.data_embedding = hstack((word_matrix, char_matrix))

    def predict(self, query_object, top_k=10):
        comb_obj = [". ".join(query_object)]
        query_vector = hstack((self.word_vectorizer.transform(comb_obj), self.char_vectorizer.transform(comb_obj)))

        similarities = cosine_similarity(query_vector, self.data_embedding)
        top_k_indices = np.argsort(similarities[0])[::-1][1:top_k + 1]

        return top_k_indices


class ObjectsTextSimilarity():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def fit(self, data):
        self.data = data
        self.name_text_features = self.data.columns
        vectors = []
        for name in self.name_text_features:
            text_feature = list((self.data.loc[:, [name]]).values.flatten())
            vectors.append(self.model.encode(text_feature, convert_to_tensor=True).to(device))

        self.data_embedding = torch.cat(vectors, dim=1)

    def predict(self, query_object_lst, top_k=10, filtr_ind=None):
        vectors = self.model.encode(query_object_lst, convert_to_tensor=True).to(device)

        query_vector = vectors.view(-1)
        similarities = torch.nn.functional.cosine_similarity(query_vector, self.data_embedding).cpu().numpy()
        # for api
        if filtr_ind is not None:
            similarities[filtr_ind] = -1

        top_k_indices = np.argsort(similarities)[::-1][1:top_k + 1]

        return top_k_indices


class ObjectsSimilarityFiltered():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def fit(self, data_text, filter_data):
        self.data_text = data_text
        self.filter_data = filter_data
        self.name_text_features = self.data_text.columns
        vectors = []
        for name in self.name_text_features:
            text_feature = list((self.data_text.loc[:, [name]]).values.flatten())
            vectors.append(self.model.encode(text_feature, convert_to_tensor=True).to(device))

        self.data_embedding = torch.cat(vectors, dim=1)

    def _filter(self, row):
        return sum(1 for col, value in zip(row.index, self.filter_features) if row[col] == value)

    def predict(self, query_object_text, filter_features, top_k=10):
        self.filter_features = filter_features
        vectors = self.model.encode(query_object_text, convert_to_tensor=True).to(device)
        query_vector = vectors.view(-1)
        similarities = torch.nn.functional.cosine_similarity(query_vector, self.data_embedding).cpu().numpy()

        similar_indices_len = len(np.where(similarities >= 0.8)[0])
        sorted_ind = np.argsort(similarities)[::-1]

        if similar_indices_len < top_k + 1:
            return sorted_ind[1:top_k + 1]

        sorted_ind_filter = sorted_ind[1:similar_indices_len]
        res = []
        num_matches = len(filter_features)
        matches = self.filter_data.iloc[sorted_ind_filter].apply(self._filter, axis=1)

        while len(res) != top_k and num_matches != -1:
            all_match_ind = list(matches.loc[matches == num_matches].index)

            if top_k - len(res) < len(all_match_ind):
                res.extend(all_match_ind[:(top_k - len(res))])
            else:
                res.extend(all_match_ind)

            num_matches -= 1

        return res
