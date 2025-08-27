# MedNLP: An Integrated Pipeline For Named Entity Recognition, Summarization And Intelligent Medical Question Answering
Named Entity Recognition • Abstractive Summarization • RAG Based Intelligent Medical Question Answering

## **Overview**

The MedNLP project is a modular NLP pipeline designed to transform raw biomedical and clinical text into structured, summarized, and queryable knowledge.
It integrates three core components:

_**1. Named Entity Recognition (NER):**_ Extracts clinically relevant entities (diseases, drugs, procedures, anatomy, lab values).

_**2. Abstractive Summarization:**_ Generates concise summaries of biomedical articles using transformer-based models.

_**3. RAG-based Medical Question Answering:**_ Provides accurate, context-grounded answers to medical questions using retrieval-augmented generation.

This pipeline supports clinical decision-making, patient engagement, and biomedical research by leveraging state-of-the-art models like SciSpacy, BART, LLaMA 3, FAISS, and MiniLM embeddings.

## **Datasets**

PubMed Article Summarization Dataset - [Link](https://www.kaggle.com/datasets/thedevastator/pubmed-article-summarization-dataset)

MTSamples Medical Transcription Dataset - [Link](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)


## **Methodology**

**1. Named Entity Recognition (NER)**

  Models: en_ner_bionlp13cg_md, en_ner_bc5cdr_md, en_core_med7_lg, biomedical-ner-all.

  Extracts clinical entities like diseases, drugs, anatomical terms, lab values, and clinical procedures etc.

**2. Summarization**

  Model: facebook/bart-large-cnn, fine-tuned on PubMed Article Summarization Dataset.

  Evaluation: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum.

**3. RAG Medical Question Answering** 

  Framework: LangChain + FAISS

  Embeddings: sentence-transformers/all-MiniLM-L6-v2

  Generator: LLaMA 3 (via Ollama)

  Evaluation: RAGAS metrics (Faithfulness, Answer Relevancy).

## **Results**

NER: Successfully identified entities like  diseases, drugs,anatomical terms, and clinical procedures etc. from unstructured transcriptions.

Summarization: Generated medically coherent and concise summaries of PubMed articles with overall scores ROUGE-1: 0.44, ROUGE-2: 0.28, ROUGE-L: 0.26, ROUGE-Lsum: 0.38.

RAG QA: Produced context-grounded answers for queries with Faithfulness ≥ 0.80 and Answer Relevancy ≥ 0.90 across most test queries.

## **User Interface**

A Gradio-based UI integrates all modules:

 Summarization Tab: Input biomedical text → get concise summary.

 NER Tab: Highlight clinical entities in transcription.

 RAG QA Tab: Ask medical questions → receive grounded answers with RAGAS scores and contexts.


## Installation
**1. Clone the repository**

git clone https://github.com/ParthaSai28/MedNLP-An-Integrated-Pipeline-For-NER-Summarization-And-Intelligent-Medical-Question-Answering.git

cd MedNLP


**2. Install dependencies**

pip install -r requirements.txt



**3. Run the Gradio interface**

python main.py -- for running RAG pipeline and document chunking

python ui.py

ollama run llama3

python -m uvicorn nlp:app --host 0.0.0.0 --port 8090 --reload

Then open your browser at http://127.0.0.1:8090


## Author

**Partha Sai Aurangabadkar**

**Toronto Metropolitan University • MSc Data Science & Analytics • 2025**
