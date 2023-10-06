import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, Birch
from torch import cosine_similarity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SimilarityPredictorKMeans:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')

    def fit(self, data):
        self.data = data
        self.kmeans.fit(data)

    def predict(self, query_object, top_k=10):
        cluster_label = self.kmeans.predict([query_object])[0]
        cluster_indices = np.where(self.kmeans.labels_ == cluster_label)[0]

        cluster_data = self.data.iloc[cluster_indices].values
        query_object_tensor = torch.tensor(query_object)
        cluster_data_tensor = torch.tensor(cluster_data)

        similarity = torch.nn.functional.cosine_similarity(query_object_tensor, cluster_data_tensor)
        similarity = similarity.numpy()

        sorted_indices = np.argsort(similarity)[::-1]
        top_k_similar_indices = cluster_indices[sorted_indices[1:top_k + 1]]

        return top_k_similar_indices


class SimilarityPredictorBIRCH:
    def __init__(self, threshold=1.7, branching_factor=100, n_clusters=None):
        self.birch = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)

    def fit(self, data):
        self.data = data
        self.birch.fit(data)

    def predict(self, query_object, top_k=10):
        cluster_label = self.birch.predict([query_object])[0]
        cluster_indices = np.where(self.birch.labels_ == cluster_label)[0]

        cluster_data = self.data.iloc[cluster_indices].values
        query_object_tensor = torch.tensor(query_object)
        cluster_data_tensor = torch.tensor(cluster_data)

        similarity = torch.nn.functional.cosine_similarity(query_object_tensor, cluster_data_tensor)
        similarity = similarity.numpy()

        sorted_indices = np.argsort(similarity)[::-1]
        top_k_similar_indices = cluster_indices[sorted_indices[1:top_k + 1]]

        return top_k_similar_indices


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
        similarities = cosine_similarity(query_vector, self.data_embedding).cpu().numpy()
        if filtr_ind is not None:
            similarities[filtr_ind] = -1
        top_k_indices = np.argsort(similarities)[::-1][1:top_k + 1]

        return top_k_indices


class ObjectsSimilarity():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def fit(self, data_features, name_text_features):
        self.data_features = data_features
        self.name_text_features = name_text_features
        vectors = []

        for i, name in enumerate(self.name_text_features):
            text_feature = list((self.data_features.loc[:, [name]]).values.flatten())
            vectors.append(self.model.encode(text_feature, convert_to_tensor=True).to(device))

        data_features_numerical = self.data_features.drop(columns=self.name_text_features).values

        self.data_embedding = torch.cat([torch.tensor(data_features_numerical).to(device)] + vectors, dim=1)

    def predict(self, query_object, top_k=10):
        vectors = [self.model.encode(query_object[name], convert_to_tensor=True).to(device) for name in
                   self.name_text_features]

        data_features_numerical = query_object.drop(labels=self.name_text_features).values.astype('float64')
        query_vector = torch.cat([torch.tensor(data_features_numerical).to(device)] + vectors, dim=0)
        similarities = cosine_similarity(query_vector, self.data_embedding).cpu().numpy()
        top_k_indices = np.argsort(similarities)[::-1][1:top_k + 1]
        return top_k_indices
