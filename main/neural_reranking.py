#this file handles reranking the top 100 results using deeplmodels

#installs
#TO DO pip install -U sentence-transformers
import torch
from transformers import BertModel, BertTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np


class NeuralRetrieval:
    def __init__(self, method="bert", doc2vec_model_path=None):
        self.method = method
        if method == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")
        elif method == "doc2vec":
            from gensim.models.doc2vec import Doc2Vec
            self.model = Doc2Vec.load(doc2vec_model_path)

    def compute_similarity(self, vec1, vec2):
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        return np.dot(vec1, vec2) / (
                    np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)  # Add small constant to avoid division by zero

    def encode(self, text):
        if self.method == "bert":
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].numpy()
        elif self.method == "doc2vec":
            return self.model.infer_vector(text.split())

    def rerank_documents(self, query, documents):
        query_vector = self.encode(query)
        doc_vectors = {doc_id: self.encode(text) for doc_id, text in doc_texts.items()}
        scores = {doc_id: self.compute_similarity(query_vector, doc_vector) for doc_id, doc_vector in
                  doc_vectors.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
