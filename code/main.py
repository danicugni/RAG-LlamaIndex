def main():

    dir_original_data = input('Enter the directory where the original data is: ')
    dir_indexed_data = input('Enter the directory where the indexed data are (if they already exist) or you would like to insert them: ')
    type_query_engine = input("Enter query engine type: ")
    query = input('Enter search query: ')
    dir_original_data = Path(dir_original_data)
    dir_indexed_data = Path(dir_indexed_data)
    index = get_index(dir_original_data= dir_original_data, dir_indexed_data= dir_indexed_data)
    print(index.ref_doc_info)
    query_engine = get_query_engine(index, type_query_engine=type_query_engine)
    response = query_engine.query(query)
    print(response)

if __name__ == "__main__":
    main()