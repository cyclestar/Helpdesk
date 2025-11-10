import os

# Folder containing your documentation
DOCS_FOLDER = "docs"

# Dictionary to store all docs
knowledge_base = {}

# Loop through all .txt files in docs/
for file_name in os.listdir(DOCS_FOLDER):
    if file_name.endswith(".txt"):
        path = os.path.join(DOCS_FOLDER, file_name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            knowledge_base[file_name] = content

# Summary
print(f"✅ Loaded {len(knowledge_base)} documents:")
for name in knowledge_base:
    print(" •", name)

# Optional: view a sample snippet
first_doc = next(iter(knowledge_base))
print(f"\n📄 Preview from {first_doc}:")
print(knowledge_base[first_doc][:300], "...")
