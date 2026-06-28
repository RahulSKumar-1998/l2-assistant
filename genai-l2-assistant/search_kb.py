import os

def search_files(directory, query):
    matches = []
    for root, dirs, files in os.walk(directory):
        if ".venv" in root or ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if query in content:
                        matches.append(filepath)
            except Exception:
                pass
    return matches

print("Searching for KB0020001...")
print(search_files(".", "KB0020001"))
