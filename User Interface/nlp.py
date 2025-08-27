import os
import re
import numpy as np
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

import torch
from datasets import Dataset
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForTokenClassification

import spacy
from retriever import load_retriever
from llm_chain import build_qa_chain

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

import ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

app = FastAPI()

# Allow your Gradio front-end to call this service from another origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- RAGAS --------------- 
ragas.llm = Ollama(model="llama3")
ragas.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faithfulness.timeout = 10000
answer_relevancy.timeout = 10000

retriever = load_retriever()
qa_chain = build_qa_chain(retriever)

# ---------- Summarization Config ----------
MODEL_DIR = r"bart_pubmed_finetuned_latest1\bart_pubmed_finetuned_latest1" 

# Tokenizer truncation length fallback
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input document text")
    min_length: int = Field(30, ge=0, description="Minimum summary length (tokens)")
    max_length: int = Field(1024, ge=1, description="Maximum summary length (tokens)")
    # Optional: if you ever want to compute ROUGE on the fly
    reference: Optional[str] = Field(None, description="Reference summary text for ROUGE")

    @validator("max_length")
    def _check_lengths(cls, v, values):
        min_len = values.get("min_length", 0)
        if v < min_len:
            raise ValueError("max_length must be >= min_length")
        return v


class SummarizeResponse(BaseModel):
    summary: str
    rouge: Dict[str, Any] = {}

_SUMM_PIPE = None
def get_summarizer():
    global _SUMM_PIPE
    if _SUMM_PIPE is not None:
        return _SUMM_PIPE

    model_path = MODEL_DIR if MODEL_DIR else MODEL_ID
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        _SUMM_PIPE = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
        )
        return _SUMM_PIPE
    except Exception as e:
        raise RuntimeError(f"Failed to load summarization model from '{model_path}': {e}")


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

# ---------- RAG API Input ----------
class Question(BaseModel):
    query: str

# ---------- NER Config ----------
SPACY_MODELS = {
    "Med7 (en_core_med7_lg)": "en_core_med7_lg",
    "BioNLP13CG (en_ner_bionlp13cg_md)": "en_ner_bionlp13cg_md",
    "BC5CDR (en_ner_bc5cdr_md)": "en_ner_bc5cdr_md",
}
HF_MODELS = {
    "HF biomedical-ner-all (d4data/biomedical-ner-all)": "d4data/biomedical-ner-all"
}

# ---------- NER Helpers ----------
class NERRequest(BaseModel):
    text: str = Field(..., min_length=1)

class Entity(BaseModel):
    start: int
    end: int
    label: str

class ModelResult(BaseModel):
    name: str
    entities: List[Entity]

class NERMultiResponse(BaseModel):
    text: str
    results: List[ModelResult]

_SPACY: Dict[str, Any] = {}
_HF: Dict[str, Any] = {}

def load_spacy_model(model_name: str):
    if model_name not in _SPACY:
        _SPACY(model_name)  # noqa

def get_spacy(model_path: str):
    if model_path not in _SPACY:
        _SPACY[model_path] = spacy.load(model_path)
    return _SPACY[model_path]

def get_hf(pipe_id: str):
    if pipe_id not in _HF:
        tok = AutoTokenizer.from_pretrained(pipe_id, use_fast=True)
        mdl = AutoModelForTokenClassification.from_pretrained(
            pipe_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        _HF[pipe_id] = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="simple",
        )
    return _HF[pipe_id]

def run_spacy_ner(nlp, text: str) -> List[Entity]:
    doc = nlp(text)
    ents: List[Entity] = []
    for e in doc.ents:
        start = int(e.start_char)
        end = int(e.end_char)
        label = str(e.label_)
        if 0 <= start < end <= len(text):
            ents.append(Entity(start=start, end=end, label=label))
    return ents

def run_hf_ner(pipe, text: str) -> List[Entity]:
    raw = pipe(text)
    ents: List[Entity] = []
    for e in raw:
        start = int(e.get("start", 0))
        end = int(e.get("end", 0))
        label = e.get("entity_group") or e.get("entity") or "ENT"
        if 0 <= start < end <= len(text):
            ents.append(Entity(start=start, end=end, label=label))
    return ents

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

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text.")

    try:
        summarizer = get_summarizer()
        tokenizer = summarizer.tokenizer
        input_length = len(tokenizer(req.text.strip()).input_ids)  # Get the number of tokens in input text
        max_output_length = min(input_length, 1024)

        # Hugging Face pipelines often prefer max_new_tokens; map UI params sensibly.
        # We'll enforce a small guard so generation doesn't fail.
        gen_kwargs = {
            "min_length": max(0, req.min_length),
            "max_length": max_output_length,
            "do_sample": False,
            "truncation": True,
        }

        # Optional: pre-truncate long inputs to keep things safe
        # Many seq2seq models handle long truncation internally, but we can be explicit
        text = req.text.strip()

        outputs = summarizer(text, **gen_kwargs)
        summary_text = outputs[0]["summary_text"] if outputs else ""

        # ROUGE: only compute if a reference was provided
        # rouge: Dict[str, Any] = {}
        # if req.reference and req.reference.strip():
        #     try:
        #         from rouge_score import rouge_scorer
        #         scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        #         scores = scorer.score(req.reference.strip(), summary_text)
        #         # Convert to simple floats (f-measure)
        #         rouge = {k: round(v.fmeasure, 4) for k, v in scores.items()}
        #     except Exception as _:
        #         # If rouge-score isn't installed or errors out, return empty dict
        #         rouge = {}
        
        # rouge_str = str(rouge)

        return SummarizeResponse(summary=summary_text)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

@app.post("/ner", response_model=NERMultiResponse)
def ner(req: NERRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")

    try:
        results: List[ModelResult] = []

        # spaCy models (exactly as in your notebook)
        for nice_name, mpath in SPACY_MODELS.items():
            nlp = get_spacy(mpath)
            ents = run_spacy_ner(nlp, text)
            results.append(ModelResult(name=nice_name, entities=ents))

        # # HF biomedical model (exactly as in your notebook)
        for nice_name, mid in HF_MODELS.items():
             pipe = get_hf(mid)
             ents = run_hf_ner(pipe, text)
             results.append(ModelResult(name=nice_name, entities=ents))

        return NERMultiResponse(text=text, results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER failed: {e}")
