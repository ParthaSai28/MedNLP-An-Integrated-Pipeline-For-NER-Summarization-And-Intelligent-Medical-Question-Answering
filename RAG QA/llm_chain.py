# llm_chain.py
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_qa_chain(retriever):
    llm = Ollama(model="llama3")  # Ensure this matches your pulled Ollama model

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful medical assistant trained to provide generalized answers from real-world clinical notes.

These clinical notes are drawn from individual patient case records. Your task is to abstract away from patient-specific language and instead provide a general summary of standard medical practices or treatments relevant to the query.

Avoid starting the response with phrases like "Based on the clinical notes provided."
Instead, give generalized, medically accurate answers suitable for a broad audience.

Focus on:
- Common procedures
- Typical treatment plans
- General symptoms and management
- Best practices in diagnosis or follow-up care

Use clear formatting:
- Bullet points or numbered steps where appropriate
- Break into short paragraphs
- Avoid long walls of text

Avoid referring to specific individuals, and answer clearly based on established clinical norms seen in the context.

If the answer is not explicitly supported by the context, respond with "I don't know."

Clinical Notes:
{context}

Question:
{question}

Generalized Answer:
"""
)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    return qa_chain
