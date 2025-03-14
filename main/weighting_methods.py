import math
import os

class base_weight:
    def __init__(self, inverted_index, doc_lengths):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.tot_docs = len(doc_lengths)
    
    def idf(self, term):
        num_corp = self.tot_docs 
        try:
            tot_contain = len(self.inverted_index[term])
        except:
            tot_contain = 0

        # natural logarithn here - change if needed
        return math.log(((num_corp - tot_contain + 0.5)/(tot_contain + 0.5)) + 1)
                        

class tf_idf(base_weight):
    def __init__(self, inverted_index, doc_lengths):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.tot_docs = len(doc_lengths)
    
    # Normalized Term Frequency
    def norm_tf(self, term, doc_id):
        freq_doc = self.inverted_index[term][doc_id]
        
        # we are normalizing based on total length of the document - therefore this is the percentage of the documents terms that are the given term
        return freq_doc/self.doc_lengths[doc_id]
    
    # TODO: Test that this is working and actually something we want to do
    # we are calculating the score of the query itself so we can build a vector that we can then use to calculate cosine similarity against the vectors for the document
    # 'query_full' is the list of tokens for the query
    def score_query(self, query_full, query_counts):
        query_vector = {}
        query_length = len(query_full)
        for token in query_full:
            tf = query_counts[token]/query_length
            idf = self.idf(token)
            query_vector[token] = tf*idf
        
        return query_vector
    
    # calculate tf-idf got a given term in a given doc
    def score_term_doc(self, term, doc_id):
        return self.norm_tf(term, doc_id)*self.idf(term)
    
    # calculate tf-idf for given doc for all query terms
    def score_doc(self, doc_id, query):
        fin_score = 0

        for token in query:
            if token in self.inverted_index and doc_id in self.inverted_index[token]:
                fin_score += self.score_term_doc(token, doc_id)
        
        return fin_score


'''First version is going to be for a general application. Looking to improve to 
"zone specific" bm25F after - https://web.stanford.edu/class/cs276/handouts/lecture12-bm25etc.pdf'''
# inherit tf_idf as parent so we have idf method (ease of maintenance)
class BM25(base_weight):
    def __init__(self, inverted_index, doc_lengths, k1 = 1.2, b = 0.75):
        self.inverted_index = inverted_index
        self.doc_lengths = doc_lengths
        self.tot_docs = len(doc_lengths)
        self.k1 = k1
        self.b = b 
        self.dl = sum(doc_lengths.values())
        self.avdl = self.dl/self.tot_docs

    def __B(self):
        return (1-self.b)+(self.b*(self.dl/self.avdl))

    def score_query(self, query_full, query_counts):
        query_vector = {}

        for token in query_full:
            tf_prime = query_counts[token]/self.__B()
            idf = self.idf(token)
            query_vector[token] = idf * (((self.k1 + 1)*tf_prime)/(self.k1 + tf_prime))
        
        return query_vector
    
    # calculate bm25 for a given term in a given doc
    def score_term_doc(self, term, doc_id):

        tf_prime = self.inverted_index[term][doc_id]/self.__B()
        idf = self.idf(term)
        return idf * (((self.k1 + 1)*tf_prime)/(self.k1 + tf_prime))
    
    # calculate bm25 for a given doc for all query terms
    def score_doc(self, doc_id, query):
        fin_score = 0

        for token in query:
            if token in self.inverted_index and doc_id in self.inverted_index[token]:
                fin_score += self.score_term_doc(token, doc_id)
        
        return fin_score







