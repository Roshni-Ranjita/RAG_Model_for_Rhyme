import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Dropdown model options
MODEL_OPTIONS = {
    "FLAN-T5 Small": "google/flan-t5-small",
    "FLAN-T5 Base": "google/flan-t5-base",
    "FLAN-T5 Large": "google/flan-t5-large"
}

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📘 Introduction", "🧸 Nursery Rhyme QA"])

selected_model_name = st.sidebar.selectbox(
    "Choose LLM model",
    options=list(MODEL_OPTIONS.keys()),
    index=0
)

@st.cache_resource(show_spinner=False)
def load_models(llm_model_name):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("text2text-generation", model=llm_model_name)
    return embedder, qa_model

@st.cache_data
def load_docs_and_embeddings(path="Data"):
    docs = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    embedder, _ = load_models(MODEL_OPTIONS[selected_model_name])
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

# --------- PAGE ROUTING ---------
if page == "📘 Introduction":
    st.title("📘 Introduction: Nursery Rhyme QA Model")
    st.markdown("""
    This app is a fun and educational demo of a **Retrieval-Augmented Generation (RAG)** system built with:

    - **Sentence-BERT** (`all-MiniLM-L6-v2`) for semantic embedding of nursery rhymes.
    - **L1 Distance Search** to retrieve the most relevant rhymes from a dataset.
    - **FLAN-T5 models** (instruction-tuned LLMs by Google) to answer user questions using retrieved context.

    ### How it Works
    1. You ask a question (e.g., *What did the dish do after the cow jumped over the moon?*)
    2. The system finds the closest matching rhymes using **vector similarity**.
    3. It passes both the **retrieved rhymes** and your **question** to an LLM to generate a natural-language answer.

    ### Select a model from the sidebar and head to the QA page to try it out! 🎉
    """)
elif page == "🧸 Nursery Rhyme QA":
    st.title("🧸 Nursery Rhyme QA Assistant")
    st.markdown("Ask a question about common nursery rhymes, and the model will try to answer it!")

    query = st.text_input("Enter your question here:")

    if query:
        embedder, qa = load_models(MODEL_OPTIONS[selected_model_name])
        docs, doc_embeddings = load_docs_and_embeddings()
        answer, dist = query_rhymes(query, docs, doc_embeddings, embedder, qa)
        st.write(f"**Answer:** {answer}")
        st.write(f"_(Distance Score: {dist:.2f})_")
# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Roshni Bhowmik")