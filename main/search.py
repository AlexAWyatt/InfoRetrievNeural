import numpy as np
from collections import defaultdict # defaultdict is an object that automatically assigns a value to key that has not been set before so that we dont get a key error
from weighting_methods import *
from utils import *
from similarity_measures import *

class SearchEngine:
    def __init__(self, model, similarity_measure = "cos_sim"):
        self.method = model
        self.results = {} #dictionaries to store vectors for cosine similarity
        self.query_counts = {}
        self.similarity_measure = similarity_measure

    def retrieve_relevant_docs(self, query_tokens):
        doc_scores = defaultdict(float) #assigning each score to the document.

        #loop through each token in the query
        for token in query_tokens:

            if token in self.method.inverted_index:
                items = self.method.inverted_index[token].items()

                #go through each doc containing this token
                for doc_id, _ in items:
                    
                    if doc_id not in doc_scores:
                        #add the token's weight to the docs total score (we only do one token at a time)
                        doc_scores[doc_id] = {}

                    # if token not yet tested - create entry for storage
                    if token not in doc_scores[doc_id]:
                        doc_scores[doc_id][token] = 0

                    # Independently find and save score of every token to allow later calculation of cosine similarity
                    doc_scores[doc_id][token] += self.method.score_term_doc(token, doc_id)

        return doc_scores

    # Queries are tokenized in main and saved
    def rank_documents(self,query_tokenized, query_num):

        #get the candiate documents that contain at least 1 query token
        doc_scores = self.retrieve_relevant_docs(query_tokenized)
        similarities = {}
        query_token_counts = self.query_counts[query_num]

        query_vector = self.method.score_query(query_tokenized, query_token_counts)

        for doc_id in doc_scores:
            store = doc_scores[doc_id]
            if self.similarity_measure == "cos_sim":
                similarities[doc_id] = calc_cos_similarity(query_vector, doc_scores[doc_id])
            elif self.similarity_measure == "raw_score":
                similarities[doc_id] = calc_raw_score(doc_scores[doc_id])

        #sort the documents in descending order
        ranked_docs = sorted(similarities.items(), key=lambda x:x[1], reverse=True)
        return ranked_docs

    # all results are stored in an object tied to the specific instance of the SearchEngine class
    def search(self, queries, run_name="my_run", num_top_docs = 100):

        for query_num, query_tokens in queries.items():

            self.query_counts[query_num] = get_token_counts(query_tokens)
            #rank the documents based on the selected method for the given query
            ranked_docs = self.rank_documents(query_tokens, query_num)
            self.results[query_num] = {doc_id: score for doc_id, score in ranked_docs[:num_top_docs]}