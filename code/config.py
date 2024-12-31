# LLM
# LLM_MODEL = 'gpt-3.5-turbo-16k'
LLM_MODEL = 'gpt-4'
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
TEMPERATURE = 0.0

# Text chunking type
SPLITTER_TYPE = 'semantic' #['sentence', 'token', 'semantic']

# Sentence or Token splitter
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20

# Semantic splitter
BUFFER_SIZE = 1
BREAKPOINT_PERCENTILE_THRESHOLD = 95

#Retriever
RETRIEVER_TYPE = 'rrf' #['vector', 'bm25', 'rrf']
STEMMER_LANGUAGE = 'italian'
SIMILARITY_TOP_K = 5
NUM_QUERIES = 2


# Query engine
RESPONSE_MODE = 'compact'
SIMILARITY_CUTOFF = 0.0
