import json

# Create dictionary keyed using document numbers where values are the 
# length of the given document in terms of tokens
def collect_doc_lengths(documents): 
    return {doc['DOCNO']:len(doc['TEXT']) for doc in documents}

def get_token_counts(query_tokens):
    token_counts = {}
    for token in query_tokens:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1
    
    return token_counts

# pair query numbers and query text into a dict for use in search function
def pair_usable_query(queries):

    usable_queries = {}

    for query in queries:
        usable_queries[query['num']] = query['query']
    
    return usable_queries

# convert output into form interpretable by trec_eval
def convert_output_form(outputs, run_name):
    results = []

    for query in outputs:

        query_id = query

        for rank, returns in enumerate(outputs[query]):

            doc_id = returns
            score = round(outputs[query][returns],7)
            
            results.append(f"{query_id} Q0 {doc_id} {rank + 1} {score} {run_name}")
    
    return results

#save the output
def save_output(main_output,path): #inverted index we want to save, path to the file location
    with open(path, 'w', encoding='utf-8') as file: #open a file at the path location
        json.dump(main_output, file, indent=4) #put the output in the file

#load a previous output that has been saved
def load_output(path): #file path
    with open(path, 'r', encoding='utf-8') as file: #open the file at the path location
        prev_output=json.load(file) #previous output is the content of the file
    return prev_output

# save output of list
def save_list_output(main_output, path):
    with open(path, 'w', encoding = 'utf-8') as file:
        for line in main_output:
            file.write(f"{line}\n")