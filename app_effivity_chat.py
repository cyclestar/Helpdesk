import streamlit as st
import os
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# ---- Load Docs ----
DOCS_FOLDER = "docs"
def load_docs():
    docs = {}
    for file_name in os.listdir(DOCS_FOLDER):
        if file_name.endswith(".txt"):
            with open(os.path.join(DOCS_FOLDER, file_name), "r", encoding="utf-8") as f:
                docs[file_name] = f.read()
    return docs

# ---- Semantic Search ----
def build_embeddings(model, docs):
    return {name: model.encode(text, convert_to_tensor=True) for name, text in docs.items()}

def semantic_search(query, model, docs, embeddings, top_k=2):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = {n: util.cos_sim(q_emb, emb).item() for n, emb in embeddings.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(n, docs[n]) for n, _ in ranked[:top_k]]

# ---- Ask AI ----
def ask_ai(query, context, history):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = [{"role": "system", "content": "You are an Effivity support assistant using internal documentation."}]
    messages += history
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"})
    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
    return res.choices[0].message.content.strip()

# ---- UI ----
st.set_page_config(page_title="Effivity AI Assistant", layout="centered")
st.title("🤖 Effivity AI Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

docs = load_docs()
model = SentenceTransformer('all-MiniLM-L6-v2')
emb = build_embeddings(model, docs)

query = st.chat_input("Ask Effivity AI...")
if query:
    matches = semantic_search(query, model, docs, emb)
    context = "\n---\n".join([f"{n}:\n{text[:800]}" for n, text in matches])
    answer = ask_ai(query, context, st.session_state.history)
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
