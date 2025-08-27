# embedding.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_vectorstore(chunks, index_path="faiss_index"):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = FAISS.from_documents(chunks, embed_model)
    vectordb.save_local(index_path)
    return vectordb