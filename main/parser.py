#inspired from the code in the assignment's example
#this section is about going through the dataset

#imports
import json
import os

#reading all the files from specific folder and parse documents from each file
def parse_documents_from_folder(folder_path):
    all_documents=[]
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            all_documents.extend(parse_documents_from_file(file_path))
    return all_documents

#taking each line and adding them to our document
def parse_document(document_line):
    doc = json.loads(document_line)

    #for each document, we reformat it so that it is in the form below
    parsed_doc = {
        'DOCNO': doc['_id'],
        'HEAD': doc.get('title','NO_TITLE'),
        'TEXT': doc.get('text','NO_TEXT'),
        'URL': doc.get('metadata',{}).get('url','NO_URL')
    }
    return parsed_doc

#going through the document and taking it one line at a time
def parse_documents_from_file(file_path):
    #we open the file
    with open(file_path,'r',encoding='utf-8') as file:
        #we save each document - processed in the current desired format - altogether.
        parsed_docs = [parse_document(line) for line in file]
    return parsed_docs

#taking each query and adding them to our document
def parse_query(query_line):
    query = json.loads(query_line)
    parsed_query = {
        'num': query['_id'],
        'query': query.get('text','NO_TEXT'),
        'evidence': query.get('metadata',{})
    }
    return parsed_query

#going through the query documents, reading the json line file and parsing each query
def parse_queries_from_file(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        parsed_queries = [parse_query(line) for line in file]
    return parsed_queries
