"""Build (or rebuild) the local Chroma vector store from documents in data/."""
import pathlib

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownTextSplitter

load_dotenv()

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PERSIST_DIR = str(ROOT / "chroma_db")
COLLECTION_NAME = "technova_docs"


def build_vector_store() -> Chroma:
    splitter = MarkdownTextSplitter(chunk_size=800, chunk_overlap=100)
    texts, metadatas = [], []

    for path in sorted(DATA_DIR.glob("*.md")):
        chunks = splitter.split_text(path.read_text(encoding="utf-8"))
        texts.extend(chunks)
        metadatas.extend({"source": path.name} for _ in chunks)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"Indexed {len(texts)} chunks from {len(list(DATA_DIR.glob('*.md')))} files into {PERSIST_DIR}")
    return store


if __name__ == "__main__":
    build_vector_store()
