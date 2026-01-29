from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import pickle
import numpy as np

# ---------------- Настройки ----------------
DATA_DIR = Path("data/knowledge_base")  # Папка с подготовленными текстами
INDEX_FILE = "data/faiss_index.bin"          # Файл для FAISS индекса
METADATA_FILE = "data/faiss_metadata.pkl"    # Файл для метаданных

CHUNK_SIZE = 500       # символы в чанке
CHUNK_OVERLAP = 50     # перекрытие
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # модель эмбеддингов

def create_index():
    # ---------------- Проверяем тексты ----------------
    files = list(DATA_DIR.glob("*.txt"))
    if not files:
        print(" Нет текстовых файлов для обработки!")
        return

    print(f"Found {len(files)} text files. Splitting into chunks...")

    # ---------------- Разделяем тексты на чанки ----------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    documents = []
    metadata = []

    for file in files:
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            print(f"⚠️ Файл {file.name} пустой, пропускаем")
            continue
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({
                "source_file": file.name,
                "chunk_id": i,
                "text": chunk
            })

    if not documents:
        print("Не удалось создать ни одного чанка!")
        return

    print(f"Total chunks created: {len(documents)}")

    # ---------------- Загружаем модель эмбеддингов ----------------
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ---------------- Генерируем эмбеддинги ----------------
    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")  # FAISS требует float32

    # ----------------  Создаём FAISS индекс ----------------
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index (dimension={dimension})...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index contains {index.ntotal} vectors")

    # ---------------- Сохраняем индекс и метаданные ----------------
    print(f"Saving FAISS index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)

    print(f"Saving metadata to {METADATA_FILE}...")
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ FAISS index and metadata saved successfully!")

# ---------------- 7️⃣ Точка входа ----------------
if __name__ == "__main__":
    create_index()
