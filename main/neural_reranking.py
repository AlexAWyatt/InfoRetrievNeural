import torch
from sentence_transformers import CrossEncoder
import logging

class CrossEncoderReranker:
    def __init__(self, corpus, model_name='cross-encoder/ms-marco-MiniLM-L6-v2', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CrossEncoder(model_name, device = self.device)
        self.corpus = corpus
        self.ranked_docs = {}
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Function to take a single query with its list of relevant documents and return a reranked list using the chosen sentence transformer model
    # relevant docs is a list of the document id's that were determined relevant to the given query by the intial info retrieval
    def rank_documents_one_q(self, parsed_query, relevant_docs):
        scores = {}

        pairs = [(parsed_query, self.corpus[doc_id]) for doc_id in relevant_docs]
        # calculate similarity score using cross encoder
        scores_tmp = self.model.predict(pairs)
        scores = {key:scores_tmp[ind] for ind, key in enumerate(relevant_docs)}

        #sort the documents in descending order
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs

    # input is a list of queries and a list of search engine outputs showing each query and their relevant documents
    def rank_all_docs(self, query_list, search_output):
        fin_res = {}

        for query_id in query_list:
            search_q = search_output[query_id]
            rel_docs = list(search_q.keys())
            q_text = query_list[query_id]
            fin_res[query_id] = {doc_id: score for doc_id, score in self.rank_documents_one_q(q_text,rel_docs)} 

        self.ranked_docs = fin_res
        return fin_res
