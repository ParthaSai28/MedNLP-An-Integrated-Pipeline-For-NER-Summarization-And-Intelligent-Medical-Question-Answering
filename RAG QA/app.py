from fastapi import FastAPI, Request
from pydantic import BaseModel
from retriever import load_retriever
from llm_chain import build_qa_chain

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
import numpy as np
import re

# Set custom LLM and embedding
import ragas

ragas.llm = Ollama(model="llama3")
ragas.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faithfulness.timeout = 10000
answer_relevancy.timeout = 10000


app = FastAPI()
retriever = load_retriever()
qa_chain = build_qa_chain(retriever)

# ---------- Text Cleaning for RAGAS ----------
def flatten_text(text):
    """
    Clean markdown and formatting for RAGAS evaluation
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # Bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)       # Italics
    text = re.sub(r"_(.*?)_", r"\1", text)         # Underscore italics
    text = re.sub(r"\n?\d+\.\s*[\w\s]+?:", '', text)  # "1. Step:"
    text = re.sub(r"^\s*[\*\-\d]+\.\s*", '', text, flags=re.MULTILINE)  # "- Bullet" or "1. Text"
    text = re.sub(r"\n[\-\*]\s*", ' ', text)       # Inline bullets
    text = text.replace('\n', ' ')                 # Newlines
    text = re.sub(r'\s+', ' ', text)               # Extra spaces
    return text.strip()

# ---------- API Input ----------
class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(data: Question):
    # Run the retrieval + LLM chain
    response = qa_chain(data.query)

    # Full output for the user
    user_output = {
        "answer": response["result"],
        "question": response["query"],
        "contexts": [doc.page_content for doc in response["source_documents"]],
        "metadata": [doc.metadata for doc in response["source_documents"]],
    }

    # Cleaned copy for evaluation only
    #cleaned = {
    #    "answer": flatten_text(response["result"]),
    #    "question": response["query"],
    #    "contexts": [flatten_text(doc.page_content) for doc in response["source_documents"]],
    #}
    
    #print("Answer for eval:", cleaned["answer"])
    #print("Context for eval:", cleaned["contexts"])

    dataset = Dataset.from_list([user_output])

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas.llm,
        embeddings=ragas.embedder
    )

    scores_dict = results.to_pandas().iloc[0].to_dict()
    safe_scores = {
        k: (0 if v is None or isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
        for k, v in scores_dict.items()
    }

    return {
        "response": user_output,
        "scores": {
            k: safe_scores[k]
            for k in ["faithfulness", "answer_relevancy"]
        }
    }
