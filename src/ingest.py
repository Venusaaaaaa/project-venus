import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use absolute paths or relative to script location for reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "sensors", "car_sensors_guide.pdf")
DB_PATH = os.path.join(BASE_DIR, "..", "db", "faiss_index")

def ingest():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print(f"Loading PDF from {DATA_PATH}...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embeddings model...")
    # using langchain_huggingface which is the modern standard
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS vectorstore...")
    vectordb = FAISS.from_documents(chunks, embeddings)

    print(f"Saving vectorstore to {DB_PATH}...")
    vectordb.save_local(DB_PATH)

    print("Ingestion complete! FAISS index created.")

if __name__ == "__main__":
    ingest()
