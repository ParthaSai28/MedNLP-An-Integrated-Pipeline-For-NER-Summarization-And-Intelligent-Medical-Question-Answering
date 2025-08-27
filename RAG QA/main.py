# main.py

from load_dataset import load_mtsamples
from chunking import chunk_documents
from embedding import build_vectorstore

if __name__ == "__main__":
    print(" Loading MTSamples dataset...")
    documents = load_mtsamples("mtsamples.csv")

    print(f" Loaded {len(documents)} documents.")

    print(" Chunking documents...")
    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} text chunks.")

    print(" Embedding & indexing with FAISS...")
    build_vectorstore(chunks, index_path="faiss_index")

    print("Vector index saved at: faiss_index/")