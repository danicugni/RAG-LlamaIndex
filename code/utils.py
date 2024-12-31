from dotenv import load_dotenv
import tiktoken
from pathlib import Path
import numpy as np

import Stemmer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser,TokenTextSplitter
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse
from llama_index.retrievers.bm25 import BM25Retriever
from config import (EMBED_MODEL, 
                    LLM_MODEL, 
                    TEMPERATURE, 
                    SPLITTER_TYPE, 
                    CHUNK_SIZE, 
                    CHUNK_OVERLAP, 
                    BUFFER_SIZE, 
                    BREAKPOINT_PERCENTILE_THRESHOLD, 
                    STEMMER_LANGUAGE, 
                    RETRIEVER_TYPE, 
                    SIMILARITY_TOP_K, 
                    NUM_QUERIES,
                    SIMILARITY_CUTOFF,
                    RESPONSE_MODE)

import nest_asyncio
nest_asyncio.apply()


def set_global_settings(embed_model:str = EMBED_MODEL, llm_model:str = LLM_MODEL, temperature:float = TEMPERATURE, splitter_type:str = SPLITTER_TYPE, 
                        chunk_size:int = CHUNK_SIZE, chunk_overlap:int = CHUNK_OVERLAP, buffer_size:int = BUFFER_SIZE, 
                        breakpoint_percentile_threshold:int = BREAKPOINT_PERCENTILE_THRESHOLD):

    load_dotenv()

    Settings.tokenizer = tiktoken.encoding_for_model(llm_model).encode

    Settings.llm = AzureOpenAI(
        engine=llm_model, 
        model=llm_model, 
        temperature=temperature
        )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name = embed_model
    )


    if splitter_type == 'sentence':
        text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'token':
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif splitter_type == 'semantic':
        text_splitter = SemanticSplitterNodeParser(buffer_size=buffer_size, breakpoint_percentile_threshold=breakpoint_percentile_threshold, embed_model=Settings.embed_model) 
    else:
        raise ValueError('Please, choose between "sentence", "token" and "semantic".')

    Settings.text_splitter = text_splitter


def create_new_index(dir_original_data:Path, dir_indexed_data:Path):
    parser = LlamaParse(result_type="markdown")
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_dir=dir_original_data, file_extractor=file_extractor, recursive=True).load_data()

    text_splitter = Settings.text_splitter

    nodes = text_splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, show_progress = True)

    index.storage_context.persist(persist_dir=dir_indexed_data)
    
    return index


def load_existing_index(dir_indexed_data:Path):

    storage_context = StorageContext.from_defaults(persist_dir=dir_indexed_data)
    index = load_index_from_storage(storage_context)
    
    return index


def get_index(dir_original_data:Path, dir_indexed_data:Path):
    
    set_global_settings()

    if dir_indexed_data.exists():
        index = load_existing_index(dir_indexed_data)
    else:
        index = create_new_index(dir_original_data, dir_indexed_data)
    
    return index


QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)


def get_retriever(index:VectorStoreIndex, similarity_top_k:int = SIMILARITY_TOP_K, num_queries:int = NUM_QUERIES,retriever_type:str = RETRIEVER_TYPE, stemmer_language:str = STEMMER_LANGUAGE):
    
    if retriever_type == 'vector':
        retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=similarity_top_k,
            )

    elif retriever_type == 'bm25':
        retriever = BM25Retriever.from_defaults(
            docstore=index.docstore, 
            similarity_top_k=similarity_top_k,
            stemmer=Stemmer.Stemmer(stemmer_language),
            language=stemmer_language
            )

    elif retriever_type == 'rrf':
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            )
        
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.docstore, 
            similarity_top_k=similarity_top_k,
            stemmer=Stemmer.Stemmer(stemmer_language),
            language=stemmer_language
            )

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=int(similarity_top_k),
            num_queries=num_queries,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            query_gen_prompt=QUERY_GEN_PROMPT,
        )
    
    else:
        raise ValueError('Please, choose between "vector", "bm25" and "rrf".')
    
    return retriever


def get_query_engine(retriever, similarity_cutoff:float = SIMILARITY_CUTOFF, response_mode:str = RESPONSE_MODE):
    
    response_synthesizer = get_response_synthesizer(response_mode=response_mode)
    
    node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=node_postprocessors,
    )

    return query_engine
