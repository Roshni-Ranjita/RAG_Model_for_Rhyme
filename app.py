# app.py
import os
import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("text2text-generation", model="google/flan-t5-small")
    return embedder, qa_model

@st.cache_data
def load_docs_and_embeddings(path="Data"):
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    embedder, _ = load_models()
    embeddings = embedder.encode(docs).astype('float32')
    return docs, embeddings

def l1_search(query_emb, doc_embeddings, top_k=3):
    dists = np.sum(np.abs(doc_embeddings - query_emb), axis=1)
    top_k_idx = np.argsort(dists)[:top_k]
    top_k_dists = dists[top_k_idx]
    return top_k_idx, top_k_dists

def query_rhymes(prompt, docs, doc_embeddings, embedder, qa, top_k=3, distance_threshold=18.0):
    query_emb = embedder.encode([prompt]).astype('float32')[0]
    top_k_idx, top_k_dists = l1_search(query_emb, doc_embeddings, top_k)

    if top_k_dists[0] > distance_threshold:
        return "I'm sorry, I couldn't find any relevant rhyme to answer that question.", top_k_dists[0]

    context = "\n".join([docs[i] for i in top_k_idx])
    full_prompt = f"Given the following nursery rhymes:\n{context}\n\nQuestion: {prompt}"
    response = qa(full_prompt, max_length=500, do_sample=False)[0]['generated_text']
    return response, top_k_dists[0]

# --------- Streamlit UI ---------
st.title("Nursery Rhyme QA Assistant 🧸🎤")
st.markdown("Ask a question about common nursery rhymes. The app will find the best match and try to answer it.")

query = st.text_input("Enter your question here:")

if query:
    embedder, qa = load_models()
    docs, doc_embeddings = load_docs_and_embeddings()
    answer, dist = query_rhymes(query, docs, doc_embeddings, embedder, qa)
    st.write(f"**Answer:** {answer}")
    st.write(f"_(Distance Score: {dist:.2f})_")
