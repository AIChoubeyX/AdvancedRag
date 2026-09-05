"""Step 4: store the vectors in a vector store for fast retrieval."""

import os

from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

# Build vector store usjng FAISS and Jina embeddings

def build_vector_store(chunks):
    """ Embed every chunk and build  searchable FAISS index in memory. """
    logger.info("Embedding %d chunk(s) and building FAISS index in memory" , len(chunks))
    embeddings_model = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    logger.info("FAISS index built in memory")
    return vector_store


# save vector store
def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH):
    """ Save the FAISS index to disk so we dont have to rebuild it every time. """
    vector_store.save_local(path)

def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    embeddings_model = get_embeddings_model()
    """ Load a previously saved FAISS index from disk. """
    return FAISS.load_local(path, embeddings_model , allow_dangerous_deserialization=True)


def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:
    """ Check if the FAISS index exists on disk. """
    return os.path.exists(os.path.join(path , "index.faiss")) 

def get_retriever(vector_store , k: int = config.TOP_K_RESULTS):
    """ Return a retriever that can be used to query the vector store. """
    logger.info("Returning retriever for vector store with top_k=%d" , k)
    return vector_store.as_retriever(search_kwargs={"k": k})