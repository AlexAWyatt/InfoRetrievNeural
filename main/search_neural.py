from sklearn.metrics.pairwise import cosine_similarity
from doc2vec_rank import *

class SearchNeural:
    def __init__(self, vector_size, epochs, min_count = 1, seed = 5):