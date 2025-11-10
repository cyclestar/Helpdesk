import os
import datetime
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# --------------------------
# 1. Load documents
# --------------------------
DOCS_FOLDER = "docs"

def load_docs():
    docs = {}
    for file_name in os.listdir(DOCS_FOLDER):
        if file_name.endswith(".txt"):
            with open(os.path.join(DOCS_FOLDER, file_name), "r", encoding="utf-8") as f:
                docs[file_name] = f.read()
    return docs

# --------------------------
# 2. Create embeddings for docs
# --------------------------
def build_doc_embeddings(model, docs):
    embeddings = {}
    for name, text in docs.items():
        embeddings[name] = model.encode(text, convert_to_tensor=True)
    return embeddings

# --------------------------
# 3. Search docs semantically
# --------------------------
def semantic_search(query, model, docs, doc_embeddings, top_k=2):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = {name: util.cos_sim(query_embedding, emb).item() for name, emb in doc_embeddings.items()}
    sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_matches = sorted_docs[:top_k]
    return [(name, docs[name]) for name, _ in top_matches]

# --------------------------
# 4. Ask Groq AI with context
# --------------------------
def ask_ai(query, context):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""
    You are an Effivity support assistant.
    Use the provided documentation context to answer the user's question clearly.

    Context:
    {context}

    Question:
    {query}

    If the context doesn't include the answer, say 'The documentation does not cover this topic.'
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --------------------------
# 5. Main logic
# --------------------------
if __name__ == "__main__":
    print("🔍 Loading Effivity documentation...")
    docs = load_docs()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = build_doc_embeddings(model, docs)
    print(f"✅ Loaded and embedded {len(docs)} documents.\n")

    while True:
        query = input("💬 Ask Effivity AI (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        matches = semantic_search(query, model, docs, doc_embeddings)
        context = "\n---\n".join([f"{name}:\n{text[:1000]}" for name, text in matches])

        answer = ask_ai(query, context)
        print(f"\n🤖 AI Answer:\n{answer}\n{'-'*80}\n")

        # Log interaction
        log_entry = (
            f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
            f"Q: {query}\n"
            f"Docs used: {', '.join([n for n, _ in matches])}\n"
            f"A: {answer}\n{'-'*80}\n"
        )
        with open("ai_history.log", "a", encoding="utf-8") as f:
            f.write(log_entry)
