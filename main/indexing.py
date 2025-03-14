#inspired by the code example provided

#imports
import json

#build the inverted index
def invert_index(proc_documents): #input the preprocessed documents here 
    #initialize
    inv_index={}
    #build the list for each document
    for doc in proc_documents: #go through each document
        #find id and content of the document
        doc_id = doc['DOCNO']
        doc_tokens = doc['TEXT'] + doc['HEAD']
        for token in doc_tokens: #now go through the contents
            #not already in the index, adds it to the index
            if token not in inv_index:
                inv_index[token]={}
            #if no token of this id have been caught, initialize
            if doc_id not in inv_index[token]:
                inv_index[token][doc_id] = 0
            #Add one to the count
            inv_index[token][doc_id] += 1
    return inv_index #we now return the inverted index

#build the inverted index using online titles
def invert_index_title(proc_documents): #input the preprocessed documents here 
    #initialize
    inv_index={}
    #build the list for each document
    for doc in proc_documents: #go through each document
        #find id and content of the document
        doc_id = doc['DOCNO']
        doc_tokens = doc['HEAD']
        for token in doc_tokens: #now go through the contents
            #not already in the index, adds it to the index
            if token not in inv_index:
                inv_index[token]={}
            #if no token of this id have been caught, initialize
            if doc_id not in inv_index[token]:
                inv_index[token][doc_id] = 0
            #Add one to the count
            inv_index[token][doc_id] += 1
    return inv_index #we now return the inverted index



#save an index in a file, in a specific location
def save_inv_index(inv_index,path): #inverted index we want to save, path to the file location
    with open(path, 'w', encoding='utf-8') as file: #open a file at the path location
        json.dump(inv_index, file, indent=4) #put the index in the file

#load the index contained in a file
def load_inv_index(path): #file path
    with open(path, 'r', encoding='utf-8') as file: #open the file at the path location
        inv_index=json.load(file) #inverted index is the content of the file
    return inv_index

""" #measure the length of the documents
def calc_docs_length(docs): #documents we want to measure
    #initialize the list of lengths
    doc_lengths={}
    #calculate the length for each doc
    for doc in docs:
        doc_id=doc['DOCNO']
        doc_tokens=doc['TEXT']
        doc_length=len(doc_tokens)
        doc_lengths[doc_id]=doc_length
    #return the lengths
    return doc_lengths """