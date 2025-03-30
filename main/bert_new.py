#inspired by https://medium.com/@Roy.Wong/step-by-step-guide-how-to-use-bert-word-embeddings-in-python-ac7b621771d8
from os.path import dirname

import nltk
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import utils
from preprocessing import dataset_dir


def tokenize_documents(parsed_docs, tokenizer):
    return [tokenizer.tokenize(" ".join(doc["TEXT"])) for doc in parsed_docs]
def tokenize(): #code taken from main

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
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    #using a set as it is easier to look up things from (in O(1) as opposed to O(n) from a list)
    stop_words1 = set(stopwords.words('english'))

    # read in StopWords List - 779 words
    stop_words2 = set()
    with open(dataset_dir + "\\StopWords.txt") as file:
        for line in file:
            stop_words2.add(line.rstrip())

    # define stopwords
    stop_words = [stop_words1, stop_words2]
    stop_words_labels = ["nltk_wrds", "ink_wrds"]

    
    # Define stemmers
    stemmers = [PorterStemmer(), LancasterStemmer(), EnglishStemmer()]
    stemmer_labels = ["porter", "lancaster", "snowball"]

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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_texts = [tokenizer.tokenize(" ".join(doc["TEXT"])) for doc in parsed_docs[0]]


#get the ids of the tokens
ids_tokens = tokenizer.convert_tokens_to_ids(tokenized_txt)

#Display the tokens
for t in zip(tokenized_txt, ids_tokens):
    print('{:<12} {:>8,}'.format(t[0], t[1]))

segments_ids = [1] * len(tokenized_txt)

token_tensor = torch.tensor([ids_tokens])
segment_tensor = torch.tensor([segments_ids])

# Load pre-trained model with the weights
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True, return_dict = False)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

#https://huggingface.co/docs/transformers/model_doc/bert#bertmodel
#The input is of the shape (batch_size, sequence_length)
#Compute the output
with torch.no_grad():
    outputs = model(token_tensor, segment_tensor)

#Concatenate all the layers
token_embeddings = torch.stack(hidden_states, dim=0)

#remove the batch dimension
token_embeddings = torch.squeeze(token_embeddings, dim=1)

token_embeddings = token_embeddings.permute(1,0,2)

token_embeddings.size()

#sum the last four layers
token_vectors_sum = []

# token_embeddings is a [35 x 13 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:

    # `token` is a [12 x 768] tensor

    # Sum the vectors from the last four layers.
    sum_vector = torch.sum(token[-4:], dim=0)

    # Use `sum_vec` to represent `token`.
    token_vectors_sum.append(sum_vector)



