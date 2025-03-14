import numpy as np
from collections import defaultdict # defaultdict is an object that automatically assigns a value to key that has not been set before so that we dont get a key error

class SearchEngine:
    def __init__(self, index, tfidf_model = None, bm25_model= None):
        self.index = index
        self.tfidf_model = tfidf_model
        self.bm25_model = bm25_model
        self.doc_vectors = {} #dictionaries to store vectors for cosine similarity

    def retrieve_relevant_docs(self, query_tokens):
        doc_scores = defaultdict(float) #assigning each score to the document.

        #loop through each token in the query
        for token in query_tokens:

            #check if token exists in the inverted index
            if token not in self.index:

                #go through each doc containing this token
                for doc_id,weight in self.index[token].items():
                    #add the token's weight to the docs total score
                    doc_scores[doc_id]+= weight

        return doc_scores

    # calculate the cosine similarity?? unsure---its a scramble of code online here and there
    def calc_cosine_similarity(self, query_tokens, doc_id):
        #initialising the query and doc vectors, the length of both vectors is the number of query tokens?
        query_vector = np.zeros(len(query_tokens))
        doc_vector = np.zeros(len(query_tokens))

        #populate
        for i,token in enumerate(query_tokens):
            #if the token is in the index and appears in the document, store its weight in the document vector
            if token in self.index and doc_id in self.index[token]:
                doc_vector[i]= self.index[token][doc_id]

            #assign a weight of 1 to the query vector for each query term
            query_vector[i] = 1

        numerator = np.dot(query_vector, doc_vector)

        #euclidean
        denominator = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)


        return numerator/denominator if denominator != 0 else 0.0

    def rank_documents(self,query, method=None):

        #process the query into tokens
        query_tokens = self.preprocess_query(query)

        #get the candiate documents that contain at least 1 query token
        doc_candidates = self.retrieve_relevant_docs(query_tokens)

        scores = {}

        for doc_id in doc_candidates:
            if method =="tfidf" and self.tfidf_model:
                scores[doc_id] = self.tfidf_model.score_doc(doc_id, query_tokens)
            elif method =="bm25" and self.bm25_model:
                scores[doc_id] = self.bm25_model.score_doc(doc_id, query_tokens)
            else:
                scores[doc_id] = self.calc_cosine_similarity(query_tokens, doc_id)
        #sort the documents in descending order
        ranked_docs = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        return ranked_docs

    def search(self, query, method=None, query_id=1, run_name="my_run"):
        #rank the documents based on the selected method
        ranked_docs = self.rank_documents(query, method)
        result_table = {}
        result_lines = []

        for rank, (doc_id, score) in enumerate(ranked_docs[:100]):  #llimit to top 100 results
            result_line = f"{query_id} Q0 {doc_id} {rank + 1} {score:.4f} {run_name}"  #formatting
            result_lines.append(result_line)
            result_table[doc_id] = score

        return result_lines, result_table