#this file handles reranking the top 100 results using deeplmodels

#installs
#TO DO pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer, util
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
class NeuralReRanker:
    def __init__(self, model_type= "bert"):
        if model_type == "bert":
            #loading a pre-trained transfomer from library
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        elif model_type == "doc2vec":
            self.model = None #will be loaded later
        self.model_type = model_type

    # training a Doc2Vec model on the provided doc
    def train_doc2vec(self,documents):
        tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]
        model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=20)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples = model.corpus_count, epochs=model.epochs)
        self.model = model

    #encoding text using selected model
    def encode(self, text):
        if self.model_type == "bert":
            #converting the input text into a dense vecto representation
            return self.model.encode(text, convert_to_tensor=True)
        elif self.model_type == "doc2vec":
            #text is split into words, then generate a fixed-sized numerical vector
            return self.model.infer_vector(text.split())

    #reranks the top_k docs using neural embeddings.
    def rerank(self, query, documents, top_k=100):
        query_embedding = self.encode(query)
        doc_embeddings = [self.encode(doc) for doc in documents]

        if self.model_type == "bert":
            scores = [util.pytorch_cos_sim(query_embedding, doc_emb).item() for doc_emb in doc_embeddings]
        elif self.model_type == "doc2vec":
            scores = [np.dot(query_embedding,doc_emb)/(np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)) for doc_emb in doc_embeddings]
