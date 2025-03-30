#most of the code here has been inspired from the assignment's example

#importing section

import os
from os.path import dirname
from parser import *
from preprocessing import *
from indexing import *
from search import *
from weighting_methods import *
from utils import *
from doc2vec_rank import *
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import EnglishStemmer

# Two setups with the most relevant documents returned are
# 1. tfidf_raw_score_ink_wrds_lancaster   308/339
# 2. tfidf_raw_score_nltk_wrds_lancaster  307/339

def main():
    # booleans to control parsing
    parse_docs = False
    parse_queries = False

    #dataset logistics
    absolute_base_path = dirname(dirname(__file__))
    dataset = absolute_base_path + "\\data\\scifact" #this is where we will change the dataset that we use
    doc_file_path = dataset + '\\corpus.jsonl'
    query_file_path = dataset + '\\queries.jsonl'
    results_file_path = absolute_base_path + "\\eval\\trec_eval-9.0.7\\test"

    # Processed files
    index_file_path = absolute_base_path + '\\data\\processed\\inverted_index.json'
    

    # Define which stopwords list to use
    # load in stopword files - 179 words
    #nltk.download('stopwords')
    #nltk.download('punkt_tab')
    #using a set as it is easier to look up things from (in O(1) as opposed to O(n) from a list)
    #stop_words1 = set(stopwords.words('english'))

    # read in StopWords List - 779 words
    stop_words2 = set()
    with open(dataset_dir + "\\StopWords.txt") as file:
        for line in file:
            stop_words2.add(line.rstrip())

    # define stopwords
    stop_words = [stop_words2]
    stop_words_labels = ["ink_wrds"]

    
    # Define stemmers
    stemmers = [LancasterStemmer()]
    stemmer_labels = ["lancaster"]

    parsed_docs = []
    parsed_queries = []
    descriptors = []

    # preprocess documents and queries for all possible combos of stop words selection and stemmers
    for stop_wordi in range(len(stop_words)):
        for stemmeri in range(len(stemmers)):
            descriptors.append(stop_words_labels[stop_wordi] + '_' + stemmer_labels[stemmeri])

            preprocessed_docs_path = absolute_base_path + '\\data\\processed\\preprocessed_docs_' + stop_words_labels[stop_wordi] + '_' + stemmer_labels[stemmeri] + '.json'
            preprocessed_queries_path = absolute_base_path + '\\data\\processed\\preprocessed_queries_' + stop_words_labels[stop_wordi] + '_' + stemmer_labels[stemmeri] + '.json'
            print(f"Parsing the dataset with stopwords = {stop_words_labels[stop_wordi]} and stemmer = {stemmer_labels[stemmeri]}...")
            documents=[]
            queries = []

            

            #preprocessing the documents
            if os.path.exists(preprocessed_docs_path) and not parse_docs:
                print("Loading preprocessed documents...")
                documents = load_preprocessed_data(preprocessed_docs_path)
            else:
                print("Preprocessing documents...")
                # change params here to use different stemmer and different stop words list / to not use either
                documents = preprocess_documents(parse_documents_from_file(doc_file_path), removestopwords=True, stopwords=stop_words[stop_wordi], stem_text=True, stemmer = stemmers[stemmeri])
                save_preprocessed_data(documents, preprocessed_docs_path)
            
            parsed_docs.append(documents)

            #Preprocessing the queries if they have not been preprocessed yet
            if os.path.exists(preprocessed_queries_path) and not parse_queries:
                print("Loading preprocessed queries...")
                queries=load_preprocessed_data(preprocessed_queries_path)
            else:
                print("Preprocessing queries...")
                queries = preprocess_queries(parse_queries_from_file(query_file_path), removestopwords=True, stopwords=stop_words[stop_wordi], stem_text=True, stemmer = stemmers[stemmeri])
                save_preprocessed_data(queries, preprocessed_queries_path)
            
            parsed_queries.append(queries)
    
    print("Done Preprocessing")

    # define similarity measures
    sim_measures = ["raw_score"]

    inverted_indices = []
    
    # loop through all preprocessed documents and create an inverted index for each
    for doc in parsed_docs:
        # build inverted index
        inverted_indices.append(invert_index(doc))
    print("Done Inverted Indices")

    outputs = []

    count = 0
    for invi in range(len(inverted_indices)):
        # define weight methods
        weight_mthds = [tf_idf(inverted_indices[invi], doc_lengths=collect_doc_lengths(parsed_docs[invi]))]
        weight_mthds_lbls = ["tfidf"]

        for mthdi in range(len(weight_mthds)):
            for sim_measure in sim_measures:
                count += 1

                search_e = SearchEngine(weight_mthds[mthdi], similarity_measure = sim_measure)
                search_e.search(pair_usable_query(parsed_queries[invi]))
                print(f"Done Search {count}")

                #convert_output_form(search_e.results, "test1").to_csv(results_file_path + "\\test_out.txt", header = None, index = None, sep = ' ')
                output = convert_output_form(search_e.results, weight_mthds_lbls[mthdi] + '_' + sim_measure + '_' + descriptors[invi])

                outputs.append(output)

                save_list_output(output, results_file_path + "\\" + weight_mthds_lbls[mthdi] + '_' + sim_measure + '_' + descriptors[invi] + ".test")

    #save_inv_index(inverted_index,path) #replace path for the path you want to save inverted index to

    #print(search_e.results)

    # create index of corpus and queries without preprocessing (for nn models)
    parsed_do = parse_documents_from_file_nn(doc_file_path)
    parsed_quer = parse_queries_from_file_nn(query_file_path)

    print(len(list(parsed_do.keys())))

    # need some hyperparameter tuning for doc2vec - some can be done here, but other must be done in the 'doc2vec_rank.py'
    vec_size = 200 # change to 300 for tests
    epochs = 50 # change to 50 for tests
    min_count = 2
    docvec = Doc2Vector(vector_size=vec_size, epochs = epochs, min_count=min_count)
    docvec.train_doc2vec(parsed_do)
    docvec.get_relevant_doc_vecs(parsed_do)

    docvec.rank_all_docs(parsed_quer, search_e.results)

    print(f"\n\n\n{docvec.ranked_docs}")
    sim_measure = "cossim"
    nn_method = "doc2vec"

    output = convert_output_form(docvec.ranked_docs, nn_method + '_' + sim_measure + '_vs' + str(vec_size) + '_epoch' + str(epochs) + '_mc' + str(min_count))
    save_list_output(output, results_file_path + "\\" + nn_method + '_' + sim_measure + '_vs' + str(vec_size) + '_epoch' + str(epochs) + '_mc' + str(min_count) + ".test")
    
if __name__ == "__main__":
    main()
