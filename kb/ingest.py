import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def embed_text(texts):
    res = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in res.data]

class OpenRouterEmbedding(Embeddings):
    def embed_documents(self, texts):
        return embed_text(texts)

    def embed_query(self, text):
        return embed_text([text])[0]

def ingest():
    folder = "./docs"
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pdf") or f.endswith(".txt")
    ]
    if not files:
        print("No documents found.")
        return

    docs = []
    for f in files:
        if f.endswith(".pdf"):
            loader = PyPDFLoader(f)
        else:
            loader = TextLoader(f)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=OpenRouterEmbedding(),
        persist_directory="./chromadb_data",
        collection_name="kb_store"
    )
    vectordb.persist()
    print("Ingestion completed.")

if __name__ == "__main__":
    ingest()
