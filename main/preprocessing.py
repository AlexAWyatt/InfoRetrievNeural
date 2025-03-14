"""TODO:
- seems like lemmatization doesn't make sense for info retrieval - stick to stemming
"""

#Most of the code here is from the assignement's example.
import json

#imports go here
import re
import os
import nltk #natural language toolkit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import EnglishStemmer

relative_dir = os.getcwd()
dataset_dir = relative_dir + "\\data"

nltk.download('stopwords')
nltk.download('punkt_tab')

#using a set as it is easier to look up things from (in O(1) as opposed to O(n) from a list)
stop_words = set(stopwords.words('english'))

#tokenising the texts using nltk's word_tokenize
def tokenize(text):
    tokens = word_tokenize(text.lower())
    return tokens

#stemming each tokens in the list of tokens sent
def stem_tokens(tokens, stemmer = PorterStemmer()):
    
    return [stemmer.stem(token) for token in tokens]

# remove stopwords through reference to relevant doc
def remove_stopwords(tokens, stopwords, verbose = False):

    no_stopwords = [token for token in tokens if token not in stopwords]

    if verbose:
        print(f"Number of stop words removed: {len(tokens) - len(no_stopwords)}")
    return no_stopwords

#Remove tokens that signify absence of data
def remove_extras(tokens, verbose = False):

    # remove tokens that were only added to signify a lack of data
    no_signifiers =  [token for token in tokens if token not in ['notitle','notext','noquery', 'nonarrative']]

    if verbose:
        print(f"Number of empty data labels removed: {len(tokens) - len(no_signifiers)}")
    return no_signifiers

#preprocessing the text by turning it into "good" tokens
def preprocess_text(text, removestopwords = True, stopwords = stop_words, stem_text = True, stemmer = PorterStemmer()):
    # remove all non letter non whitespace characters using regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = tokenize(text)

    tokens = remove_extras(tokens)

    # If stemming is enabled, stem the tokens with the chosen stemmer
    if stem_text:
        tokens = stem_tokens(tokens, stemmer)
    
    # If stopword removal is enabled, do so
    if removestopwords:
        tokens = remove_stopwords(tokens, stopwords)
    return tokens

#preprocessing all of the documents now
def preprocess_documents(documents, removestopwords = True, stopwords = stop_words, stem_text = True, stemmer = PorterStemmer()):
    previous_id = 'x'
    count = 1

    #going through each document line by line and extracting information to tokenise it and put it back where it belongs.
    for doc in documents:
        file_id = str(doc['DOCNO'].split(" ")[0])
        if file_id != previous_id:
            previous_id = file_id
            count += 1
        doc['TEXT'] = preprocess_text(doc['TEXT'], removestopwords, stopwords, stem_text, stemmer)
        doc['HEAD'] = preprocess_text(doc['HEAD'], removestopwords, stopwords, stem_text, stemmer)

    return documents

#preprocessing queries
def preprocess_queries(queries, removestopwords = True, stopwords = stop_words, stem_text = True, stemmer = PorterStemmer()):
    for query in queries:
        # The only thing that needs processing for queries - there is no other usable text
        query['query'] = preprocess_text(query['query'], removestopwords, stopwords, stem_text, stemmer)
    return queries

def save_preprocessed_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def load_preprocessed_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data