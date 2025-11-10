import os

DOCS_FOLDER = "docs"

def load_docs():
    docs = {}
    for file_name in os.listdir(DOCS_FOLDER):
        if file_name.endswith(".txt"):
            with open(os.path.join(DOCS_FOLDER, file_name), "r", encoding="utf-8") as f:
                docs[file_name] = f.read()
    return docs

def search_docs(query, docs):
    results = []
    for name, content in docs.items():
        if query.lower() in content.lower():
            snippet_start = content.lower().find(query.lower())
            snippet = content[max(0, snippet_start-80):snippet_start+300]
            results.append((name, snippet))
    return results

if __name__ == "__main__":
    docs = load_docs()
    print(f"✅ Loaded {len(docs)} documents.\n")

    while True:
        query = input("🔍 Enter a question or keyword (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        matches = search_docs(query, docs)
        if not matches:
            print("⚠️ No match found in documentation.\n")
        else:
            print(f"📚 Found {len(matches)} relevant section(s):\n")
            for name, snippet in matches:
                print(f"📄 {name} →\n{snippet.strip()}\n{'-'*60}\n")
