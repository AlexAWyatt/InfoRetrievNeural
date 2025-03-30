#this file handles reranking the top 100 results using deeplmodels

#installs
#TO DO pip install -U sentence-transformers
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial import distance


class BERTReRanker:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.corp_vecs = {}
        self.ranked_docs = {}

    def get_doc_vec(self, doc):
        inputs = self.tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    def get_relevant_doc_vecs(self, corpus):
        vec_dict = {}
        for doc_id, text in corpus.items():
            vec_dict[doc_id] = self.get_doc_vec(text)
        self.corp_vecs = vec_dict
        return vec_dict

    def rank_documents_one_q(self, parsed_query, relevant_docs):
        similarities = {}
        query_vec = self.get_doc_vec(parsed_query)

        for doc_id in relevant_docs:
            similarities[doc_id] = distance.cosine(query_vec, self.corp_vecs[doc_id])

        ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs

    def rank_all_docs(self, query_list, search_output):
        fin_res = {}

        for query_id, q_text in query_list.items():
            rel_docs = search_output[
                query_id] if query_id in search_output else []  # Get the list of relevant docs for the query
            fin_res[query_id] = self.rank_documents_one_q(q_text, rel_docs)

        self.ranked_docs = fin_res
        return fin_res
