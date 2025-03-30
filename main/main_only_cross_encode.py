# most of the code here has been inspired from the assignment's example

# importing section

import os
import json
from os.path import dirname
from parser import *
from preprocessing import *
from indexing import *
from search import *
from weighting_methods import *
from utils import *
from doc2vec_rank import *
from neural_reranking import *
import torch
from sentence_transformers import CrossEncoder
import logging

def rank_documents_one_q(parsed_query, relevant_docs, corpus, model):
        scores = {}

        pairs = [(parsed_query, corpus[doc_id]) for doc_id in relevant_docs]
        # calculate similarity score using cross encoder
        scores_tmp = model.predict(pairs)
        scores = {key:scores_tmp[ind] for ind, key in enumerate(relevant_docs)}

        # TODO - MUST CHECK SCORES RETURNED AND IF THIS ORDERING IS WORKING CORRECTLY
        #sort the documents in descending order
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[0:100]


def main():

    #dataset logistics
    absolute_base_path = dirname(dirname(__file__))
    dataset = absolute_base_path + "\\data\\scifact" #this is where we will change the dataset that we use
    doc_file_path = dataset + '\\corpus.jsonl'
    query_file_path = dataset + '\\queries.jsonl'
    results_file_path = absolute_base_path + "\\eval\\trec_eval-9.0.7\\test"

    #print(search_e.results)
    parsed_do = parse_documents_from_file_nn(doc_file_path)
    parsed_quer = parse_queries_from_file_nn(query_file_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name='cross-encoder/ms-marco-MiniLM-L6-v2'
    CrossEncoder(model_name, device = device)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    nn_rank = CrossEncoder(model_name=model_name, device = None)

    fin_res = {}
    """ test = {}
    for i in list(parsed_quer.keys())[0:5]:
        test.update({i:parsed_quer[i]}) """

    for query_id in parsed_quer:
        q_text = parsed_quer[query_id]
        fin_res[query_id] = {doc_id: score for doc_id, score in rank_documents_one_q(q_text,list(parsed_do.keys()), parsed_do, nn_rank)} 

    sim_measure = "cross_encode"
    nn_method = "ms-marco-MiniLM-L6-v2"

    output = convert_output_form(fin_res, nn_method + '_' + sim_measure + "_noprerank")
    save_list_output(output, results_file_path + "\\" + nn_method + '_' + sim_measure + "_noprerank" + ".test")


if __name__ == "__main__":
    main()
