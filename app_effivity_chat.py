import os
os.system("pip install groq>=0.6.0")  # ensures groq installs on Streamlit Cloud

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# --------------------------
# 1. Page Configuration + Branding
# --------------------------
st.set_page_config(page_title="Effivity Helpdesk AI", page_icon="💡", layout="centered")

# Logo + Title + Intro
st.image("assets/effivity_logo.png", width=180)
st.markdown(
    """
    ### Welcome to **Effivity Helpdesk AI** 💬  
    Your smart assistant to find answers instantly from Effivity documentation, SOPs, and guides.  
    Just type your question below to get started!
    """
)
st.divider()

# --------------------------
# 2. Load Documents
# --------------------------
DOCS_FOLDER = "docs"

@st.cache_resource
def load_docs():
    docs = {}
    for file_name in os.listdir(DOCS_FOLDER):
        if file_name.endswith(".txt"):
            with open(os.path.join(DOCS_FOLDER, file_name), "r", encoding="utf-8") as f:
                docs[file_name] = f.read()
    return docs

docs = load_docs()

# --------------------------
# 3. Build Embeddings (Semantic Search)
# --------------------------
@st.cache_resource
def build_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = {name: model.encode(text, convert_to_tensor=True) for name, text in docs.items()}
    return model, embeddings

model, embeddings = build_embeddings()

# --------------------------
# 4. Semantic Search Function
# --------------------------
def semantic_search(query, model, docs, embeddings, top_k=2):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = {n: util.cos_sim(q_emb, emb).item() for n, emb in embeddings.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(n, docs[n]) for n, _ in ranked[:top_k]]

# --------------------------
# 5. Ask Groq AI
# --------------------------
def ask_ai(query, context, history):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = [{"role": "system", "content": "You are an Effivity support assistant using internal documentation."}]
    messages += history
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"})
    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
    return res.choices[0].message.content.strip()

# --------------------------
# 6. Chat Interface
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask Effivity Helpdesk AI...")

if query:
    matches = semantic_search(query, model, docs, embeddings)
    context = "\n---\n".join([f"{n}:\n{text[:800]}" for n, text in matches])
    answer = ask_ai(query, context, st.session_state.history)
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
