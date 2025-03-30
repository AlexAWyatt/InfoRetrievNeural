#this file handles reranking the top 100 results using deeplmodels

#installs
#TO DO pip install -U sentence-transformers
import torch
from sentence_transformers import SentenceTransformer, util

class PretrainedReRanker:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=None, doc_vec = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device = self.device)
        if doc_vec is not None:
            self.corp_vecs = doc_vec
        else:
            self.corp_vecs = {}
        self.ranked_docs = {}

    # return a document vector for a given document (in string form)
    def get_doc_vec(self, doc):
        return self.model.encode(doc, convert_to_tensor=True)

    # input 'corpus' is a dictionary of documents where key is the document id
    # get vecs for all of corpus and save based on id
    def get_doc_vecs(self, corpus):
        vec_dict = {}
        for doc_id, text in corpus.items():
            vec_dict[doc_id] = self.get_doc_vec(text)
        self.corp_vecs = vec_dict
        return vec_dict

    # Function to take a single query with its list of relevant documents and return a reranked list using the chosen sentence transformer model
    # relevant docs is a list of the document id's that were determined relevant to the given query by the intial info retrieval
    def rank_documents_one_q(self, parsed_query, relevant_docs):
        similarities = {}
        query_vec = self.get_doc_vec(parsed_query)

        for doc_id in relevant_docs:
            # calculate cosine similarity and save as numeric
            similarities[doc_id] = util.pytorch_cos_sim(query_vec, self.corp_vecs[doc_id]).item()

        # TODO - MUST CHECK SCORES RETURNED AND IF THIS ORDERING IS WORKING CORRECTLY
        #sort the documents in descending order
        ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
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
