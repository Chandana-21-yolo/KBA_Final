import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

load_dotenv()

print("OPENROUTER_API_KEY loaded:", os.getenv("OPENROUTER_API_KEY") is not None)

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

class KBAgent:
    def __init__(self, chroma_dir="./chromadb_data", collection_name="kb_store"):
        from ingest import OpenRouterEmbedding
        self.vectordb = Chroma(
            persist_directory=chroma_dir,
            collection_name=collection_name,
            embedding_function=OpenRouterEmbedding()
        )

    def answer(self, query):
        docs = self.vectordb.similarity_search(query, k=1)
        context = "\n\n".join([d.page_content for d in docs])
        if len(context) > 1000:
            context = context[:1000] + "\n\n[Truncated]"
        prompt = f"""
You are a KnowledgeBase assistant. Use ONLY the context to answer.
Context:
{context}
Question:
{query}
"""
        result = client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return result.choices[0].message.content.strip()

def get_agent():
    return KBAgent()
