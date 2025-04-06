import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# Load rhyme text files
def load_rhyme_texts(path="./rhymes"):
    texts = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

docs = load_rhyme_texts()

# Embed using SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(docs).astype('float32')
 
# Index with FAISS
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Basic QA model (you can switch to RAG if you want)
qa = pipeline("text2text-generation", model="google/flan-t5-base")

# Query
def query_rhymes(prompt, top_k=3):
    query_emb = embedder.encode([prompt]).astype('float32')
    _, indices = index.search(query_emb, top_k)
    context = "\n".join([docs[i] for i in indices[0]])
    full_prompt = f"Given the following nursery rhymes:\n{context}\n\nQuestion: {prompt}"
    response = qa(full_prompt, max_length=500, do_sample=False)[0]['generated_text']
    return response

if __name__ == "__main__":
    while True:
        query = input("Ask about a rhyme (or press Enter to exit): ")
        if not query.strip():
            break
        result = query_rhymes(query)
        print("Answer:", result)
