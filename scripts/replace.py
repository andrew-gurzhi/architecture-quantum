from pathlib import Path
import json

# Папка с чистыми текстами
IN_DIR = Path("data/clean")
OUT_DIR = Path("data/knowledge_base")
OUT_DIR.mkdir(exist_ok=True)

# Загружаем словарь замен из JSON
TERMS_FILE = Path("data/terms_map.json")
with TERMS_FILE.open("r", encoding="utf-8") as f:
    REPLACEMENTS = json.load(f)

def replace_terms(text: str) -> str:
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text

def main():
    for file in IN_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        new_text = replace_terms(text)
        out_file = OUT_DIR / file.name
        out_file.write_text(new_text, encoding="utf-8")
        print(f"Processed {file.name} → {out_file.name}")

if __name__ == "__main__":
    main()
