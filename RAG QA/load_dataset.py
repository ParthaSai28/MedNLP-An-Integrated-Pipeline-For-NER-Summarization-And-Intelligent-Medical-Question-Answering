# load_dataset.py
import pandas as pd
from langchain.docstore.document import Document

def load_mtsamples(path='mtsamples.csv'):
    df = pd.read_csv(path)
    df = df.dropna(subset=['transcription'])  # keep all, just drop missing

    docs = []
    for _, row in df.iterrows():
        metadata = {
            "medical_specialty": row["medical_specialty"],
            "sample_name": row["sample_name"]
        }
        content = row["description"] + "\n\n" + row["transcription"]
        docs.append(Document(page_content=content, metadata=metadata))

    return docs
