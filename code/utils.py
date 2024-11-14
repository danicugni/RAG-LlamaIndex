from dotenv import load_dotenv
import tiktoken
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse

import nest_asyncio
nest_asyncio.apply()


def load_existing_index(dir_indexed_data:Path):

    storage_context = StorageContext.from_defaults(persist_dir=dir_indexed_data)
    index = load_index_from_storage(storage_context)
    
    return index


def create_new_index(dir_original_data:Path, dir_indexed_data:Path):
    
    parser = LlamaParse(result_type="markdown")
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_dir=dir_original_data, file_extractor=file_extractor, recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents=documents, show_progress=True)
    index.storage_context.persist(persist_dir=dir_indexed_data)
    
    return index


def query_engine_high_level(index:VectorStoreIndex):
    
    return index.as_query_engine()


def query_engine_low_level(index:VectorStoreIndex, similarity_top_k:int = 5, similarity_cutoff:float = 0.0, response_mode:str = 'compact'):

    retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )

    response_synthesizer = get_response_synthesizer(response_mode=response_mode)
    
    node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )

    return query_engine


def set_global_settings(embed_model:str = 'BAAI/bge-small-en-v1.5', tokenizer:str = 'gpt-3.5-turbo', temperature:float = 0.0, splitter_type:str = 'sentence', chunk_size:int = 512, chunk_overlap:int = 20):

    load_dotenv()

    Settings.tokenizer = tiktoken.encoding_for_model(tokenizer).encode

    Settings.llm = AzureOpenAI(
        engine='gpt-35-turbo-16k', 
        model='gpt-35-turbo-16k', 
        temperature=temperature
        )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name = embed_model
    )

    if splitter_type == 'sentence':
        text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'token':
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else: 
        raise ValueError('Please, choose between "sentence" and "token".')

    Settings.text_splitter = text_splitter


def get_index(dir_original_data:Path, dir_indexed_data:Path):
    
    set_global_settings()

    if dir_indexed_data.exists():
        index = load_existing_index(dir_indexed_data)

    else:
        index = create_new_index(dir_original_data, dir_indexed_data)
    
    return index
    

def get_query_engine(index:VectorStoreIndex, similarity_top_k:int = 5, similarity_cutoff:float = 0.0, response_mode:str = 'compact', type_query_engine:str = 'high-level'):

    if type_query_engine == 'high-level':
        query_engine = query_engine_high_level(index)

    elif type_query_engine == 'low-level':
        query_engine = query_engine_low_level(index, similarity_top_k, similarity_cutoff, response_mode)

    else:
        raise ValueError('Please, choose between "high-level" and "low-level".')

    return query_engine


