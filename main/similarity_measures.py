import numpy as np

## TODO: ADD OTHER SIMILARITY MEASURES

# calculate the cosine similarity
def calc_cos_similarity(query_scores, doc_scores):
    #initialising the query and doc vectors, the length of both vectors is the number of query tokens?
    query_vector = np.zeros(len(query_scores))
    doc_vector = np.zeros(len(query_scores))

    #populate
    for i,token in enumerate(query_scores.keys()):
        #if the token is in the index and appears in the document, store its weight in the document vector
        if token in doc_scores:
            doc_vector[i] = doc_scores[token]
        else:
            doc_vector[i] = 0

        #assign a weight of 1 to the query vector for each query term
        query_vector[i] = query_scores[token]

    numerator = np.dot(query_vector, doc_vector)

    #euclidean
    denominator = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)

    return numerator/denominator if denominator != 0 else 0.0

# TODO: check that this is working
def calc_raw_score(doc_scores):
    score = 0
    for token in doc_scores:
        score += doc_scores[token]
    
    return score