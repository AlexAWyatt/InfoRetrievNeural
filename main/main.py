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
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import EnglishStemmer
from neural_reranking import NeuralRetrieval
from bert_new import *

# Two setups with the most relevant documents returned are
# 1. tfidf_raw_score_ink_wrds_lancaster   308/339
# 2. tfidf_raw_score_nltk_wrds_lancaster  307/339


def main():
    # booleans to control parsing
    parse_docs = False
    parse_queries = False

    #dataset logistics
    retrieval = NeuralRetrieval(doc2vec_model_path='doc2vec.model')
    absolute_base_path = dirname(dirname(__file__))
    dataset = absolute_base_path + "\\data\\scifact" #this is where we will change the dataset that we use
    doc_file_path = dataset + '\\corpus.jsonl'
    query_file_path = dataset + '\\queries.jsonl'
    results_file_path = absolute_base_path + "\\eval\\trec_eval-9.0.7\\test"

    # Processed files
    index_file_path = absolute_base_path + '\\data\\processed\\inverted_index.json'
    
    # do not remove stopwords for neural net reranking, but we will remove to get our top results.
    # So we need to match the doc and just get the stemmed version for reranking
    # Define which stopwords list to use
    # load in stopword files - 179 words

    #nltk.download('stopwords')
    #nltk.download('punkt_tab')

    #using a set as it is easier to look up things from (in O(1) as opposed to O(n) from a list)
    
    #stop_words1 = set(stopwords.words('english'))
    '''
    # read in StopWords List - 779 words
    stop_words2 = set()
    with open(dataset_dir + "\\StopWords.txt") as file:
        for line in file:
            stop_words2.add(line.rstrip())
    '''
    stop_words2 = set()
    with open("C:\\Users\\khesw\\OneDrive\\Desktop\\Winter 2025\\CSI 4107\\this assignment\\InfoRetrievNeural\\data\\StopWords.txt") as file:
        for line in file:
            stop_words2.add(line.rstrip())

    # define stopwords
    #stop_words = [stop_words1, stop_words2]
    #stop_words_labels = ["nltk_wrds", "ink_wrds"]

    # define stopwords
    stop_words = [stop_words2]
    stop_words_labels = ["ink_wrds"]

    
    # Define stemmers
    #stemmers = [PorterStemmer(), LancasterStemmer(), EnglishStemmer()]
    #stemmer_labels = ["porter", "lancaster", "snowball"]

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

            # Apply BERT Tokenization to parsed documents
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokenized_docs = [tokenize_documents(docs, tokenizer) for docs in parsed_docs]

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

    tokenized_docs = [tokenize_documents(docs, tokenizer) for docs in parsed_docs]

    # define similarity measures
    sim_measures = ["raw_score"]

    inverted_indices = [invert_index(doc) for doc in parsed_docs]

    use_neural_reranking = True
    reranker_model = "bert"
    neural_reranker = NeuralRetrieval(method=reranker_model)

    outputs = []
    count = 0
    for invi in range(len(inverted_indices)):
        # define weight methods
        #weight_mthds = [tf_idf(inverted_indices[invi], doc_lengths=collect_doc_lengths(parsed_docs[invi])), BM25(inverted_indices[invi], doc_lengths=collect_doc_lengths(parsed_docs[invi]))]
        #weight_mthds_lbls = ["tfidf", "bm25"]

        # define weight methods
        weight_mthds = [tf_idf(inverted_indices[invi], doc_lengths=collect_doc_lengths(parsed_docs[invi]))]
        weight_mthds_lbls = ["tfidf"]

        for mthdi in range(len(weight_mthds)):
            for sim_measure in sim_measures:
                count += 1

                search_e = SearchEngine(weight_mthds[mthdi], similarity_measure=sim_measure,use_nn=use_neural_reranking, reranker=neural_reranker)
                search_e.search(pair_usable_query(parsed_queries[invi]),doc_id_to_text={doc['DOCNO']: " ".join(doc['TEXT']) for doc in parsed_docs[invi]})
                print(f"Done Search {count}")

                #convert_output_form(search_e.results, "test1").to_csv(results_file_path + "\\test_out.txt", header = None, index = None, sep = ' ')
                output = convert_output_form(search_e.results, "title_only_" + weight_mthds_lbls[mthdi] + '_' + sim_measure + '_' + descriptors[invi])

                outputs.append(output)

                save_list_output(output, results_file_path + "\\title_only_" + weight_mthds_lbls[mthdi] + '_' + sim_measure + '_' + descriptors[invi] + ".test")

    #save_inv_index(inverted_index,path) #replace path for the path you want to save inverted index to

    #neural reranking integration

    #create mapping of docs of IDs
    doc_id_to_text = {doc['DOCNO']: " ".join(doc['TEXT']) for doc in parsed_docs[0]}

    #initialise, set to false to disable neural reranking
    use_neural_reranking = True
    reranker_model = "bert" #here is where we can change to "doc2vec"

    neural_reranker = NeuralRetrieval(method=reranker_model,doc2vec_model_path='doc2vec.model' if reranker_model == "doc2vec" else None)
    reranked_results = {}
    for query_id, doc_scores in search_e.results.items():
        top_doc_ids = list(doc_scores.keys())[:100]
        top_doc_texts = {doc_id: doc_id_to_text[doc_id] for doc_id in top_doc_ids}
        reranked_list = neural_reranker.rerank_documents(pair_usable_query(parsed_queries[0])[query_id], top_doc_texts)
        reranked_results[query_id] = {doc_id: score for doc_id, score in reranked_list}

    save_output(reranked_results, results_file_path + "\neural_results.json")
    print("Neural Re-ranking Completed and Results Saved!")
if __name__ == "__main__":
    main()
