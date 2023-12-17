import requests
import json
import time
from urllib.parse import quote_plus

def get_document_id(solr_url, query):
    """ Get the ID of the document matching the query. """
    encoded_query = quote_plus(query)
    full_url = f"{solr_url}?q={encoded_query}"
    response = requests.get(full_url)

    if response.status_code == 200:
        response_json = response.json()
        docs = response_json.get('response', {}).get('docs', [])
        if docs:
            return docs[0].get('id')  # Assuming the first document is the correct one
        else:
            print("No documents found for the query.")
            return None
    else:
        print(f"Error during search: {response.status_code}")
        return None

def find_similar_documents(solr_mlt_url, document_id):
    """ Find documents similar to the one with the provided ID. """
    full_url = f"{solr_mlt_url}?q=id:{quote_plus(document_id)}&mlt.fl=body&mlt.boost=true&mlt.interestingTerms=details&mlt.maxdfpct=10&ros=10"
    response = requests.get(full_url)
    print(response)

    # for run in range(100):
    if response.status_code == 200:
        print("save file")
        return response.text  # Return the raw JSON response text

    return None

def main():
    # Define the base URL for Solr
    solr_url = 'http://localhost:8990/solr/liang/select'
    solr_mlt_url = 'http://localhost:8990/solr/liang/mlt'  # URL for MLT queries

    # Query to find the specific document by title
    query = 'title:"Media Outlet Criticized for Trying to Normalize \'the Death of Trump\'"'

    # Get the document ID
    document_id = get_document_id(solr_url, query)

    if document_id:
        print(document_id)
        similar_documents_json = find_similar_documents(solr_mlt_url, document_id)

        if similar_documents_json:
            # Write the raw JSON response to a file
            with open('M1_ouput.json', 'w') as file:
                file.write(similar_documents_json)
    
    return similar_documents_json

if __name__ == "__main__":
    for i in range(10):
        out = main()
        if out is not None:
            print("success")
            break
        time.sleep(0.5)
