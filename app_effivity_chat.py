import os
os.system("pip install groq>=0.6.0")

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# --------------------------
# 1. Page Setup
# --------------------------
st.set_page_config(page_title="Effivity Helpdesk AI", page_icon="💡", layout="centered")

# Custom CSS for clean UI
st.markdown(
    """
    <style>
        .fixed-header {
            position: sticky;
            top: 0;
            background-color: #f0f2f6;
            padding: 12px;
            border-bottom: 2px solid #ddd;
            text-align: center;
            z-index: 999;
        }
        .chat-box {
            border-radius: 12px;
            padding: 10px 14px;
            margin: 6px 0;
            max-width: 85%;
        }
        .user-msg {
            background-color: #DCF8C6;
            align-self: flex-end;
            margin-left: auto;
        }
        .ai-msg {
            background-color: #FFFFFF;
            border: 1px solid #e6e6e6;
            align-self: flex-start;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# 2. Fixed Header
# --------------------------
st.markdown(
    """
    <div class="fixed-header">
        <img src="https://raw.githubusercontent.com/cyclestar/Helpdesk/main/assets/effivity_logo.png" width="120">
        <h3>Effivity Helpdesk AI 🤖</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# 3. Load Documents
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
# 4. Build Embeddings
# --------------------------
@st.cache_resource
def build_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = {name: model.encode(text, convert_to_tensor=True) for name, text in docs.items()}
    return model, embeddings

model, embeddings = build_embeddings()

# --------------------------
# 5. Semantic Search
# --------------------------
def semantic_search(query, model, docs, embeddings, top_k=2):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = {n: util.cos_sim(q_emb, emb).item() for n, emb in embeddings.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(n, docs[n]) for n, _ in ranked[:top_k]]

# --------------------------
# 6. Ask Groq AI
# --------------------------
def ask_ai(query, context, history):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = [{"role": "system", "content": "You are an Effivity support assistant using product documentation."}]
    messages += history
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"})
    res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
    return res.choices[0].message.content.strip()

# --------------------------
# 7. Chat Interface
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Welcome message
if not st.session_state.history:
    st.chat_message("assistant").markdown(
        "👋 Hi, I’m your **Effivity Helpdesk AI Assistant**. Ask me anything about Effivity features, workflows, or setup."
    )

query = st.chat_input("Type your question here...")

if query:
    matches = semantic_search(query, model, docs, embeddings)
    context = "\n---\n".join([f"{n}:\n{text[:800]}" for n, text in matches])
    answer = ask_ai(query, context, st.session_state.history)

    # Add to chat history
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})

# Display chat messages
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-box user-msg'>🧑‍💼 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-box ai-msg'>🤖 {msg['content']}</div>", unsafe_allow_html=True)
