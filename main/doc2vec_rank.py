from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import logging


class Doc2Vector:
    def __init__(self, vector_size, epochs, min_count = 1, seed = 5):
        self.model = Doc2Vec(vector_size=vector_size, min_count=min_count, seed = seed, dbow_words=1)
        self.epochs = epochs
        self.corp_vecs = {}
        self.ranked_docs = {}
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # input data is a dictionary with a key which is the document id and a value which is all the text for said id
    def train_doc2vec(self, data):
        tagged_data = []
        for doc_id in data:
            tagged_data.append(TaggedDocument(words = word_tokenize(data[doc_id].lower()),
                                              tags = doc_id))

        print("\nBuilding Doc2Vec Vocabulary")
        self.model.build_vocab(tagged_data)
        print("\nTraining Doc2Vec")
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.epochs)
    
    # return a document vector for a given document (in string form)
    def get_doc_vec(self, doc):
        return self.model.infer_vector(word_tokenize(doc.lower()))
    
    # input 'corpus' is a dictionary of documents where key is the document id
    # get vecs for all of corpus and save based on id
    def get_doc_vecs(self, corpus):
        vec_dict = dict()

        for doc_id in corpus:
            vec_dict[doc_id] = self.get_doc_vec(corpus[doc_id])

        self.corp_vecs = vec_dict
        # return a dictionary of the doc2vec vector of each document in corpus indexed by their id
        return vec_dict
    

    # Function to take a single query with its list of relevant documents and return a reranked list using doc2vec
    # relevant docs is a list of the document id's that were determined relevant to the given query by the intial info retrieval
    def rank_documents_one_q(self,parsed_query, relevant_docs):

        similarities = {}
        query_vec = self.get_doc_vec(parsed_query)

        for doc_id in relevant_docs:
            similarities[doc_id] = distance.cosine(query_vec, self.corp_vecs[doc_id])

        # TODO - MUST CHECK SCORES RETURNED AND IF THIS ORDERING IS WORKING CORRECTLY
        #sort the documents in descending order
        ranked_docs = sorted(similarities.items(), key=lambda x:x[1], reverse=True)
        return ranked_docs
    
    # input is a list of queries and a list of search engine outputs showing each query and their relevant documents
    def rank_all_docs(self, query_list, search_output):
        fin_res = dict()

        for query_id in query_list:
            search_q = search_output[query_id]
            rel_docs = list(search_q.keys())
            q_text = query_list[query_id]
            fin_res[query_id] = {doc_id: score for doc_id, score in self.rank_documents_one_q(q_text,rel_docs)} 
        
        self.ranked_docs = fin_res
        return fin_res
