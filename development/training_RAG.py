from constants import *
import fitz
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import faiss

def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text for better RAG retrieval.
    """

    # fix broken lines inside sentences
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # remove extra newlines
    text = re.sub(r"\n+", "\n", text)

    # remove weird bullet symbols
    text = re.sub(r"[•●▪▫►■]", "", text)

    # remove multiple dots
    text = re.sub(r"\.{2,}", ".", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    # normalize quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    # remove empty lines
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    return text

doc = fitz.open("data\HP_book_all.pdf")

text = ""
for page in doc:
    text += page.get_text()

text = clean_text(text)

# print(text[:1000])


from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                          chunk_overlap=200)

chunks = splitter.split_text(text)

print(len(chunks))
# print(chunks[0])


model = SentenceTransformer(EMBEDDINGS_MODEL)
embeddings = model.encode(chunks, normalize_embeddings=True)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings))

# Save chunks
with open("embeddings/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Save FAISS index
faiss.write_index(index, "embeddings/faiss_index.bin")

print("✅ Saved chunks and FAISS index")