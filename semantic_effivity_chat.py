import os
import datetime
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# --------------------------
# 1. Load documentation
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
# 2. Build semantic embeddings
# --------------------------
def build_doc_embeddings(model, docs):
    embeddings = {}
    for name, text in docs.items():
        embeddings[name] = model.encode(text, convert_to_tensor=True)
    return embeddings

# --------------------------
# 3. Semantic search
# --------------------------
def semantic_search(query, model, docs, doc_embeddings, top_k=2):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = {name: util.cos_sim(query_embedding, emb).item() for name, emb in doc_embeddings.items()}
    sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_matches = sorted_docs[:top_k]
    return [(name, docs[name]) for name, _ in top_matches]

# --------------------------
# 4. Ask Groq AI with conversation context
# --------------------------
def ask_ai(client, chat_history, query, context):
    system_prompt = (
        "You are an Effivity support assistant. "
        "Use the provided documentation context and prior conversation to help the user clearly and accurately. "
        "If something isn't in the docs, say so honestly."
    )

    messages = [{"role": "system", "content": system_prompt}] + chat_history
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# --------------------------
# 5. Main interaction loop
# --------------------------
if __name__ == "__main__":
    print("🧠 Loading Effivity documentation...")
    docs = load_docs()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = build_doc_embeddings(model, docs)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print(f"✅ Loaded and embedded {len(docs)} documents.\n")

    chat_history = []

    while True:
        query = input("💬 You: ")
        if query.lower() == "exit":
            break

        matches = semantic_search(query, model, docs, doc_embeddings)
        context = "\n---\n".join([f"{name}:\n{text[:800]}" for name, text in matches])
        answer = ask_ai(client, chat_history, query, context)

        print(f"\n🤖 AI: {answer}\n{'-'*80}\n")

        # Update conversation memory
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})

        # Limit memory to last 6 exchanges
        if len(chat_history) > 12:
            chat_history = chat_history[-12:]

        # Log everything
        with open("ai_history.log", "a", encoding="utf-8") as f:
            f.write(
                f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
                f"Q: {query}\nA: {answer}\n{'-'*80}\n"
            )
