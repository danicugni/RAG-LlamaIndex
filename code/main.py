from pathlib import Path
from utils import get_index, get_query_engine, get_retriever

def main():

    dir_original_data = input('Enter the directory where the original data is: ')
    dir_indexed_data = input('Enter the directory where the indexed data is (if it already exist) or you would like to insert it: ')
    dir_original_data = Path(dir_original_data)
    dir_indexed_data = Path(dir_indexed_data)
    query = input('Enter search query: ')
    index = get_index(dir_original_data=dir_original_data, dir_indexed_data=dir_indexed_data)
    retriever = get_retriever(index)
    query_engine = get_query_engine(retriever)
    response = query_engine.query(query)
    print(response)

if __name__ == "__main__":
    main()