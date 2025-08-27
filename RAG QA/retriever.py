# retriever.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_retriever(index_path="faiss_index"):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(index_path, embed_model, allow_dangerous_deserialization=True)
    return vectordb.as_retriever(search_kwargs={"k": 8})
