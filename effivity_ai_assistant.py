import os
import datetime
from groq import Groq

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
# 2. Simple keyword search
# --------------------------
def search_docs(query, docs):
    results = []
    for name, content in docs.items():
        if query.lower() in content.lower():
            snippet_start = content.lower().find(query.lower())
            snippet = content[max(0, snippet_start-80):snippet_start+500]
            results.append((name, snippet))
    return results

# --------------------------
# 3. Ask Groq AI using context
# --------------------------
def ask_ai(query, context):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
    You are an Effivity support assistant.
    Use the provided documentation context to answer the user's question clearly and concisely.

    Context:
    {context}

    Question:
    {query}

    If the context doesn't have the answer, say 'The documentation does not cover this topic.'
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --------------------------
# 4. Main interaction with logging
# --------------------------
if __name__ == "__main__":
    docs = load_docs()
    print(f"✅ Loaded {len(docs)} documents.\n")

    while True:
        query = input("💬 Ask Effivity AI (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        matches = search_docs(query, docs)
        if not matches:
            print("⚠️ No relevant info found in docs.\n")
            continue

        combined_context = "\n---\n".join([f"{n}:\n{s}" for n, s in matches[:2]])
        answer = ask_ai(query, combined_context)

        print(f"\n🤖 AI Answer:\n{answer}\n{'-'*80}\n")

        # Log interaction
        log_entry = (
            f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
            f"Q: {query}\n"
            f"Context used: {', '.join([n for n, _ in matches[:2]])}\n"
            f"A: {answer}\n"
            f"{'-'*80}\n"
        )

        with open("ai_history.log", "a", encoding="utf-8") as f:
            f.write(log_entry)
